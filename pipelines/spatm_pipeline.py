import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, TextualInversionLoaderMixin
# diffusers renamed LoraLoaderMixin -> StableDiffusionLoraLoaderMixin around v0.27.
# Import whichever exists so this works across diffusers versions on Lambda Stack.
try:
    from diffusers.loaders import StableDiffusionLoraLoaderMixin as LoraLoaderMixin
except ImportError:
    from diffusers.loaders import LoraLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
# adjust_lora_scale_text_encoder moved between diffusers versions and is NOT
# actually called in this file — guard the import so a missing path can't break load.
try:
    from diffusers.models.lora import adjust_lora_scale_text_encoder
except ImportError:
    try:
        from diffusers.utils import adjust_lora_scale_text_encoder
    except ImportError:
        adjust_lora_scale_text_encoder = None
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, logging, replace_example_docstring
# scale_lora_layers / unscale_lora_layers are imported for parity with the stock
# pipeline but are not called here; guard them so older/newer diffusers both load.
try:
    from diffusers.utils import scale_lora_layers, unscale_lora_layers
except ImportError:
    scale_lora_layers = unscale_lora_layers = None
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# _create_4d_causal_attention_mask is used in the SPATM text-encoding path. Its
# location/signature has shifted across transformers versions, so fall back to a
# local implementation if the import is unavailable.
try:
    from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
except ImportError:
    def _create_4d_causal_attention_mask(input_shape, dtype, device, *args, **kwargs):
        bsz, seq_len = input_shape
        mask = torch.full(
            (seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype
        )
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveTokenMapping_v2(nn.Module):
    """
    Concept-aware adaptive token mapping.
    Input: [placeholder || concept] concatenated = 2*input_dim
    Output: transformed placeholder of input_dim
    """

    def __init__(self, input_dim=768, hidden_dim=1024, dtype=torch.float16):
        super().__init__()
        self.input_dim = input_dim
        # Input is [placeholder || concept] = 2 * input_dim
        self.mapping = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.dtype = dtype
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        x: [N, input_dim] — x[0]=placeholder, x[1:]=concept token(s)
        Returns: [1, input_dim] — transformed placeholder
        """
        target_dtype = self.mapping[0].weight.dtype
        orig_dtype = x.dtype
        x = x.to(dtype=target_dtype)

        placeholder = x[0:1]                              # [1, 768]
        concept = x[1:].mean(dim=0, keepdim=True)        # [1, 768]
        combined = torch.cat([placeholder, concept], dim=-1)  # [1, 1536]

        out = self.mapping(combined)                      # [1, 768]
        return (placeholder + out).to(dtype=orig_dtype)  # residual


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class StableDiffusionAdaptiveTokenPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        adaptive_mapping: AdaptiveTokenMapping_v2,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            adaptive_mapping=adaptive_mapping,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.use_peft_backend = False  # Add missing use_peft_backend attribute
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds
    
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        lora_scale=None,
        clip_skip=None,
        profession_name=None,
        token_name=None,
    ):
        
        print("\n===== ENCODE PROMPT DEBUG =====")
        print("prompt =", prompt)
        print("profession_name =", profession_name)
        print("token_name =", token_name)
        print("token_name type =", type(token_name))
        print("==============================\n")

        # =====================================================
        # PROMPT FORMAT
        # =====================================================

        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)

        # =====================================================
        # TOKENIZATION
        # =====================================================

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)

        print("\n===== TOKEN DEBUG =====")

        for i, token_id in enumerate(text_input_ids[0]):

            token = self.tokenizer.convert_ids_to_tokens(
                token_id.item()
            )

            print(
                f"{i:2d} | {token} | id={token_id.item()}"
            )

            if token == "<|endoftext|>":
                break

        print("=======================\n")

        attention_mask = (
            text_inputs.attention_mask.to(device)
            if hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
            else None
        )

        # =====================================================
        # ORIGINAL TOKEN EMBEDDINGS
        # =====================================================

        token_embeddings = self.text_encoder.get_input_embeddings()(
            text_input_ids
        )

        # =====================================================
        # SPATM PLACEHOLDER TOKEN
        # =====================================================

        if token_name is not None and profession_name is not None:

            placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
                token_name
            )

            print("placeholder_token_id =", placeholder_token_id)
            print(
                "matches =",
                (text_input_ids[0] == placeholder_token_id).sum().item()
            )

            placeholders_mask = (
                text_input_ids == placeholder_token_id
            )
            print(
                "Placeholder positions:",
                torch.where(placeholders_mask)[1].tolist()
            )

            print(
                "placeholder_token_id:",
                placeholder_token_id
            )

            print(
                "matches:",
                placeholders_mask.sum().item()
            )

            print(
                "placeholder_token_id:",
                placeholder_token_id
            )

            print(
                "matches:",
                (text_input_ids == placeholder_token_id)
                .sum()
                .item()
            )

        else:

            placeholders_mask = torch.zeros_like(
                text_input_ids
            ).bool()

        # =====================================================
        # APPLY SPATM
        # =====================================================

        if placeholders_mask.any() and self.adaptive_mapping is not None:

            print("=== SPATM ACTIVE ===")

            # =================================================
            # ORIGINAL CLIP TOKEN EMBEDDINGS
            # =================================================

            original_embeddings = token_embeddings[
                placeholders_mask
            ]

            # =================================================
            # SPATM MAPPING OUTPUT
            # =================================================

            if self.adaptive_mapping is not None:

                # Build concept embedding for concept-aware mapping
                if profession_name is not None:
                    concept_ids = set(
                        self.tokenizer.encode(profession_name, add_special_tokens=False)
                    )
                    concept_mask = torch.zeros_like(text_input_ids).bool()
                    for idx, tid in enumerate(text_input_ids[0]):
                        if tid.item() in concept_ids:
                            concept_mask[0, idx] = True
                    concept_embeddings = token_embeddings[concept_mask]
                    if concept_embeddings.shape[0] == 0:
                        concept_embeddings = original_embeddings
                else:
                    concept_embeddings = original_embeddings

                # Pass [placeholder, concept] — mapping handles concatenation internally
                input_for_mapping = torch.cat(
                    [original_embeddings, concept_embeddings], dim=0
                )
                adaptive_output = self.adaptive_mapping(input_for_mapping)  # [1, 768]

                print("\n===== MAPPING OUTPUT DEBUG =====")

                raw_delta = adaptive_output - original_embeddings

                print("raw delta norm:",
                    raw_delta.norm(dim=-1).mean().item())

                print("raw delta max:",
                    raw_delta.abs().max().item())

                print("raw delta mean:",
                    raw_delta.abs().mean().item())

                print("===============================\n")

            else:

                print("=== SPATM DISABLED ===")

                adaptive_output = original_embeddings
            # =================================================
            # DEBUG: CHECK HOW MUCH SPATM CHANGED THE EMBEDDING
            # =================================================

            cos_sim = F.cosine_similarity(
                original_embeddings,
                adaptive_output,
                dim=-1
            )

            print(
                "Cosine similarity:",
                cos_sim.mean().item()
            )

            print(
                "Mean abs delta:",
                (adaptive_output - original_embeddings)
                .abs()
                .mean()
                .item()
            )

            print(
                "Original norm:",
                original_embeddings.norm(dim=-1).mean().item()
            )

            print(
                "Mapped norm:",
                adaptive_output.norm(dim=-1).mean().item()
            )

            print("\n===== EMBEDDING VALUES =====")

            print("Original embedding first 10 dims:")
            print(original_embeddings[0][:10])

            print("\nMapped embedding first 10 dims:")
            print(adaptive_output[0][:10])

            print("\nDifference first 10 dims:")
            print((adaptive_output - original_embeddings)[0][:10])

            print("============================\n")

            print(
                "Max abs delta:",
                (adaptive_output - original_embeddings)
                .abs()
                .max()
                .item()
            )

            # ==========================================
            # DEBUG CHECKS
            # ==========================================

            print("\n===== SPATM DEBUG =====")

            print(
                "placeholder count:",
                placeholders_mask.sum().item()
            )

            print(
                "original norm:",
                original_embeddings.norm(dim=-1).mean().item()
            )

            print(
                "adaptive norm:",
                adaptive_output.norm(dim=-1).mean().item()
            )

            print(
                "adaptive/original ratio:",
                (
                    adaptive_output.norm(dim=-1).mean()
                    /
                    original_embeddings.norm(dim=-1).mean()
                ).item()
            )

            print("========================\n")

            # =================================================
            # HYPERSPHERE STABILIZATION
            # =================================================

            scale_factor = 1.0

            # Preserve original CLIP magnitude

            original_norm = torch.norm(
                original_embeddings,
                dim=-1,
                keepdim=True
            )

            # Normalize both embeddings

            normalized_original = torch.nn.functional.normalize(
                original_embeddings,
                dim=-1
            )

            normalized_adaptive = torch.nn.functional.normalize(
                adaptive_output,
                dim=-1
            )

            # Safe directional blend

            blended_direction = (
                (1.0 - scale_factor) * normalized_original
                + scale_factor * normalized_adaptive
            )

            # Restore original embedding energy

            blended_embeddings = (
                torch.nn.functional.normalize(
                    blended_direction,
                    dim=-1
                ) * original_norm
            )

            print("\n===== BLEND DEBUG =====")

            print("original norm:",
                original_embeddings.norm(dim=-1).mean().item())

            print("adaptive norm:",
                adaptive_output.norm(dim=-1).mean().item())

            print("blended norm:",
                blended_embeddings.norm(dim=-1).mean().item())

            blend_cos = F.cosine_similarity(
                original_embeddings,
                blended_embeddings,
                dim=-1
            )

            print("original -> blended cosine:",
                blend_cos.mean().item())

            print("=======================\n")
        
            # =================================================
            # SAFE UPDATE
            # =================================================

            token_embeddings[
                placeholders_mask
            ] = blended_embeddings

        else:

            print("=== BASELINE MODE ===")


        # =====================================================
        # BASELINE PATH
        # =====================================================

        if (not placeholders_mask.any()) or self.adaptive_mapping is None:

            print("=== PURE SD1.5 BASELINE ===")

            prompt_embeds = self.text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]

        # =====================================================
        # SPATM PATH
        # =====================================================

        else:

            print("=== SPATM ACTIVE ===")

            # =====================================================
            # POSITION IDS
            # =====================================================

            position_ids = self.text_encoder.text_model.embeddings.position_ids[
                :, : token_embeddings.shape[1]
            ]

            # =====================================================
            # POSITION EMBEDDINGS
            # =====================================================

            position_embeddings = (
                self.text_encoder.text_model.embeddings.position_embedding(
                    position_ids
                )
            )

            # =====================================================
            # FINAL HIDDEN STATES
            # =====================================================

            hidden_states = token_embeddings + position_embeddings

            # =====================================================
            # CLIP TRANSFORMER ENCODER
            # =====================================================

            input_shape = text_input_ids.shape   # (batch_size, seq_len)
            causal_attention_mask = _create_4d_causal_attention_mask(
                input_shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )

            encoder_outputs = self.text_encoder.text_model.encoder(
                inputs_embeds=hidden_states,
                attention_mask=None,
                causal_attention_mask=causal_attention_mask,   
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

            # =====================================================
            # LAST HIDDEN STATE
            # =====================================================

            prompt_embeds = encoder_outputs.last_hidden_state

            # =====================================================
            # FINAL LAYER NORM
            # =====================================================

            prompt_embeds = self.text_encoder.text_model.final_layer_norm(
                prompt_embeds
            )

        # =====================================================
        # DTYPE
        # =====================================================

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(
            dtype=prompt_embeds_dtype,
            device=device
        )

        # =====================================================
        # DUPLICATE FOR MULTIPLE IMAGES
        # =====================================================

        bs_embed, seq_len, _ = prompt_embeds.shape

        prompt_embeds = prompt_embeds.repeat(
            1,
            num_images_per_prompt,
            1
        )

        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt,
            seq_len,
            -1
        )

        # =====================================================
        # NEGATIVE PROMPT
        # =====================================================

        if do_classifier_free_guidance:

            uncond_tokens = (
                [negative_prompt]
                if isinstance(negative_prompt, str)
                else [""] * batch_size
            )

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=seq_len,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)

            uncond_attention_mask = (
                uncond_input.attention_mask.to(device)
                if hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
                else None
            )

            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids,
                attention_mask=uncond_attention_mask,
            )[0]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype,
                device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1,
                num_images_per_prompt,
                1
            )

            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt,
                seq_len,
                -1
            )

        return prompt_embeds, negative_prompt_embeds
    
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        profession_name: Optional[str] = None,
        token_name: Optional[str] = None
        # change_step: int = 0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # device = self._execution_device
        device = self.unet.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
            profession_name=profession_name,
            token_name=token_name
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        
        ######below code is commented for SPATM, which requires switching the prompt embeddings in the middle of the denoising process. We will prepare two sets of prompt embeddings and switch them at the specified step in the denoising loop.
        # if do_classifier_free_guidance:
        #     # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        #     inclusive_prompt_embeds = torch.cat([negative_prompt_embeds[0].unsqueeze(0), prompt_embeds[0].unsqueeze(0)])
        #     ori_prompt_embeds = torch.cat([negative_prompt_embeds[1].unsqueeze(0), prompt_embeds[1].unsqueeze(0)])
        #     print(f'inclusive_prompt_embeds: {inclusive_prompt_embeds.shape}')
        #     print(f'ori_prompt_embeds: {ori_prompt_embeds.shape}')
        # raise NotImplementedError
        ################################################################
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # prompt_embeds = ori_prompt_embeds if i < change_step else inclusive_prompt_embeds
                # expand the latents if we are doing classifier free guidance
                # =====================================================
                # SPATM USES SINGLE PROMPT
                # =====================================================

                latent_model_input = (
                torch.cat([latents] * 2)
                if do_classifier_free_guidance
                else latents
                )

                latent_model_input = self.scheduler.scale_model_input(
                latent_model_input,
                t
                )

                if do_classifier_free_guidance:
                    encoder_hidden_states = torch.cat(
                    [negative_prompt_embeds, prompt_embeds]
                    )
                else:
                    encoder_hidden_states = prompt_embeds

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                    )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)