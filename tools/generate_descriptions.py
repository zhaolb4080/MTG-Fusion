from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any


TEXT_KEYS = ("t_a", "t_b", "t_ab", "t_as", "t_bs", "t_gt")
DEFAULT_BLIP_CAPTION_REVISION = "a0ad1e4f8184ed95d6f80053778ca5b33aa31f5f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate six MTG-Fusion descriptions")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--backend",
        choices=("template", "blip-local", "qwen-local", "openai"),
        default="template",
    )
    parser.add_argument("--task", choices=("vis-ir", "mif"), default="vis-ir")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument(
        "--local-model", default="Salesforce/blip-image-captioning-base"
    )
    parser.add_argument(
        "--local-revision",
        default=DEFAULT_BLIP_CAPTION_REVISION,
        help="Model revision containing safetensors weights",
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--qwen-retries",
        type=int,
        default=3,
        help="Number of attempts when Qwen returns incomplete or invalid JSON",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def data_url(path: str) -> str:
    file_path = Path(path)
    mime = mimetypes.guess_type(file_path.name)[0] or "image/png"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def template_descriptions(task: str) -> dict[str, str]:
    if task == "mif":
        return {
            "t_a": "A medical image showing modality-specific anatomical or functional information.",
            "t_b": "A registered medical image showing complementary tissue structures and boundaries.",
            "t_ab": "Both images depict the same aligned anatomy, lesions, tissue regions, and spatial layout.",
            "t_as": "Preserve clinically useful structures and responses that are distinctive in modality A.",
            "t_bs": "Preserve clinically useful tissue contrast and boundaries distinctive in modality B.",
            "t_gt": "A clear fused medical image retaining complementary anatomy, functional signals, boundaries, and faithful color without artifacts.",
        }
    return {
        "t_a": "An infrared image emphasizing thermal targets and heat responses in the scene.",
        "t_b": "A visible image preserving illumination, natural structures, edges, and textures.",
        "t_ab": "Both images depict the same aligned scene, objects, boundaries, and spatial layout.",
        "t_as": "Preserve distinctive infrared thermal saliency and bright heat-emitting targets.",
        "t_bs": "Preserve distinctive visible-spectrum textures, fine edges, and background structure.",
        "t_gt": "A clear fused image with salient thermal targets, faithful visible details, balanced contrast, and no artifacts.",
    }


def local_descriptions(caption_a: str, caption_b: str, task: str) -> dict[str, str]:
    caption_a = caption_a.strip().rstrip(".")
    caption_b = caption_b.strip().rstrip(".")
    if task == "mif":
        return {
            "t_a": f"Modality A shows {caption_a}.",
            "t_b": f"Modality B shows {caption_b}.",
            "t_ab": (
                f"The registered medical views share the same anatomy and spatial layout: "
                f"modality A shows {caption_a}, while modality B shows {caption_b}."
            ),
            "t_as": (
                f"Preserve the clinically useful responses and structures specific to "
                f"modality A, where {caption_a}."
            ),
            "t_bs": (
                f"Preserve the tissue contrast and boundaries specific to modality B, "
                f"where {caption_b}."
            ),
            "t_gt": (
                f"Produce a clear fused medical image combining the complementary evidence "
                f"from {caption_a} and {caption_b}, with faithful boundaries and no artifacts."
            ),
        }
    return {
        "t_a": (
            f"The infrared image shows {caption_a} and emphasizes thermal intensity and "
            f"heat-emitting targets."
        ),
        "t_b": (
            f"The visible image shows {caption_b} and preserves natural structures, "
            f"illumination, edges, and textures."
        ),
        "t_ab": (
            f"The registered infrared and visible images depict the same scene and spatial "
            f"layout: the infrared view shows {caption_a}, while the visible view shows "
            f"{caption_b}."
        ),
        "t_as": (
            f"Preserve infrared-specific thermal saliency and bright heat responses in the "
            f"scene containing {caption_a}."
        ),
        "t_bs": (
            f"Preserve visible-specific background structure, fine edges, and textures in "
            f"the scene containing {caption_b}."
        ),
        "t_gt": (
            f"Produce a clear fused image of {caption_b}, retaining thermal targets from "
            f"{caption_a}, visible details, balanced contrast, and no artifacts."
        ),
    }


class LocalBlipCaptioner:
    def __init__(self, model_name: str, revision: str, device_name: str) -> None:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        use_cuda = device_name.startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device(device_name if use_cuda else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = BlipProcessor.from_pretrained(
            model_name, revision=revision, use_fast=False
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            revision=revision,
            use_safetensors=True,
            dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def __call__(self, image_paths: list[Path]) -> list[str]:
        import torch
        from PIL import Image

        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            output_ids = self.model.generate(
                pixel_values=pixel_values,
                max_new_tokens=30,
                num_beams=3,
            )
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)


def parse_description_json(response: str) -> dict[str, str]:
    start = response.find("{")
    if start < 0:
        raise ValueError(f"Local VLM returned no JSON object: {response[:200]}")
    values, _ = json.JSONDecoder().raw_decode(response[start:])
    if not isinstance(values, dict):
        raise TypeError("Local VLM output must be a JSON object")
    missing = [key for key in TEXT_KEYS if not isinstance(values.get(key), str)]
    if missing:
        raise KeyError(f"Local VLM output is missing string fields: {missing}")
    return {key: values[key].strip() for key in TEXT_KEYS}


def recover_description_json(response: str, task: str) -> dict[str, str]:
    """Recover valid fields from incomplete Qwen output and safely fill the rest."""
    recovered: dict[str, str] = {}
    start = response.find("{")
    if start >= 0:
        try:
            values, _ = json.JSONDecoder().raw_decode(response[start:])
            if isinstance(values, dict):
                recovered.update(
                    {
                        key: value.strip()
                        for key, value in values.items()
                        if key in TEXT_KEYS
                        and isinstance(value, str)
                        and value.strip()
                    }
                )
        except (TypeError, ValueError):
            # A response cut off by max_new_tokens may not be a complete JSON object.
            pass

    # Preserve complete JSON string fields even if the surrounding object is truncated.
    for key in TEXT_KEYS:
        if key in recovered:
            continue
        match = re.search(
            rf'"{re.escape(key)}"\s*:\s*("(?:\\.|[^"\\])*")', response
        )
        if match is not None:
            try:
                value = json.loads(match.group(1)).strip()
            except (AttributeError, TypeError, ValueError):
                continue
            if value:
                recovered[key] = value

    fallback = template_descriptions(task)
    return {key: recovered.get(key, fallback[key]) for key in TEXT_KEYS}


class LocalQwenDescriber:
    def __init__(
        self,
        model_name: str,
        device_name: str,
        max_new_tokens: int,
        retries: int,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        use_cuda = device_name.startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device(device_name if use_cuda else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            use_safetensors=True,
            dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.retries = max(1, retries)

    @staticmethod
    def prompt(task: str) -> str:
        if task == "mif":
            modality = (
                "The first and second images are registered medical images from "
                "different modalities."
            )
            specific_a = "clinically useful information specific to modality A"
            specific_b = "clinically useful information specific to modality B"
        else:
            modality = "The first image is infrared and the second image is visible."
            specific_a = "infrared-specific thermal targets and salient heat responses"
            specific_b = "visible-specific illumination, textures, edges, and background detail"
        return (
            modality
            + " Compare both images carefully and return only one valid JSON object with "
            "exactly six English string fields. t_a: an image-specific description of the "
            "first image. t_b: an image-specific description of the second image. t_ab: "
            "shared scene, objects, structures, and spatial correspondence. t_as: "
            + specific_a
            + ". t_bs: "
            + specific_b
            + ". t_gt: a concrete description of the ideal fused image that preserves "
            "complementary information without artifacts. Keep every field concise, "
            "distinguish shared from modality-specific content, and do not invent objects. "
            "Every key is mandatory; never omit a field. Use exactly this schema: "
            '{"t_a":"...","t_b":"...","t_ab":"...","t_as":"...",'
            '"t_bs":"...","t_gt":"..."}'
        )

    def _generate(
        self,
        image_a: Path,
        image_b: Path,
        prompt_text: str,
        max_new_tokens: int,
    ) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_a.resolve().as_uri()},
                    {"type": "image", "image": image_b.resolve().as_uri()},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def __call__(self, image_a: Path, image_b: Path, task: str) -> dict[str, str]:
        last_response = ""
        last_error: Exception | None = None
        base_prompt = self.prompt(task)
        for attempt in range(self.retries):
            prompt_text = base_prompt
            if attempt:
                prompt_text += (
                    " This is a format-correction retry. The previous response was invalid. "
                    "Return all six keys, including t_gt, as one complete JSON object and "
                    "write no text before or after it."
                )
            last_response = self._generate(
                image_a,
                image_b,
                prompt_text,
                self.max_new_tokens + 128 * attempt,
            )
            try:
                return parse_description_json(last_response)
            except (KeyError, TypeError, ValueError) as error:
                last_error = error
                print(
                    f"warning: invalid Qwen JSON for {image_a.name} "
                    f"(attempt {attempt + 1}/{self.retries}): {error}"
                )

        recovered = recover_description_json(last_response, task)
        print(
            f"warning: Qwen JSON retries exhausted for {image_a.name}; "
            f"using recovered fields plus safe defaults ({last_error})"
        )
        return recovered


def resolve_image_path(value: str, manifest: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def openai_descriptions(
    client: Any, model: str, image_a: str, image_b: str, task: str
) -> dict[str, str]:
    from pydantic import BaseModel

    class DescriptionBundle(BaseModel):
        t_a: str
        t_b: str
        t_ab: str
        t_as: str
        t_bs: str
        t_gt: str

    modality = (
        "Image A is infrared and image B is visible."
        if task == "vis-ir"
        else "Images A and B are registered medical images from different modalities."
    )
    prompt = (
        modality + " Return only a JSON object with "
        "six string fields: t_a describes A; t_b describes B; t_ab describes shared "
        "scene/object/structure information; t_as describes infrared-specific useful "
        "information; t_bs describes visible-specific useful information; t_gt describes "
        "the ideal fused image. Be image-specific, concise, and do not invent objects."
    )
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url(image_a)},
                    {"type": "input_image", "image_url": data_url(image_b)},
                ],
            }
        ],
        text_format=DescriptionBundle,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("The model returned no structured description output")
    values = parsed.model_dump()
    return {key: values[key] for key in TEXT_KEYS}


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    input_path = Path(args.input).resolve()
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.resume and output.exists():
        completed = {
            str(json.loads(line)["id"])
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    pending = [row for row in rows if str(row["id"]) not in completed]
    mode = "a" if args.resume and output.exists() else "w"

    client = None
    if args.backend == "openai":
        from openai import OpenAI

        client = OpenAI()
    captioner = (
        LocalBlipCaptioner(args.local_model, args.local_revision, args.device)
        if args.backend == "blip-local"
        else None
    )
    qwen_describer = (
        LocalQwenDescriber(
            args.qwen_model,
            args.device,
            args.max_new_tokens,
            args.qwen_retries,
        )
        if args.backend == "qwen-local"
        else None
    )

    with output.open(mode, encoding="utf-8") as destination:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            if captioner is not None:
                image_paths: list[Path] = []
                for row in batch:
                    image_paths.extend(
                        (
                            resolve_image_path(str(row["ir"]), input_path),
                            resolve_image_path(str(row["vis"]), input_path),
                        )
                    )
                captions = captioner(image_paths)
            for index, row in enumerate(batch):
                if captioner is not None:
                    descriptions = local_descriptions(
                        captions[index * 2], captions[index * 2 + 1], args.task
                    )
                elif qwen_describer is not None:
                    descriptions = qwen_describer(
                        resolve_image_path(str(row["ir"]), input_path),
                        resolve_image_path(str(row["vis"]), input_path),
                        args.task,
                    )
                elif client is not None:
                    descriptions = openai_descriptions(
                        client,
                        args.model,
                        str(resolve_image_path(str(row["ir"]), input_path)),
                        str(resolve_image_path(str(row["vis"]), input_path)),
                        args.task,
                    )
                else:
                    descriptions = template_descriptions(args.task)
                row.update(descriptions)
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                destination.flush()
            print(f"completed={len(completed) + min(start + len(batch), len(pending))}/{len(rows)}")
    print(f"wrote descriptions to {output}")


if __name__ == "__main__":
    main()
