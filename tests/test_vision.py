import base64
import io

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from pydantic import ValidationError

from fashion_agent.vision import (
    ItemRecognition,
    LookChange,
    LookCritique,
    ReferenceReading,
    VisionCallFailed,
    VisionClient,
    VisionUnavailable,
    get_vision_client,
    normalize_image,
    reset_vision_client,
)

JPEG = b"\xff\xd8\xff\xe0fakejpegbytes"


def jpeg_of(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (30, 60, 90)).save(buffer, "JPEG")

    return buffer.getvalue()


class StubStructured:
    def __init__(self, value):
        self.value = value

    def invoke(self, messages):
        self.messages = messages

        return self.value


class StubLLM:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def with_structured_output(self, schema):
        self.calls.append(schema)

        return StubStructured(self.value)


def critique(**overrides) -> LookCritique:
    scores = {
        "occasion_fit": 8,
        "cohesion": 8,
        "colour_harmony": 8,
        "proportions": 8,
        "silhouette": 8,
        "summary": "ок",
    }

    scores.update(overrides)

    return LookCritique(**scores)


def client(value) -> VisionClient:
    return VisionClient(llm=StubLLM(value))


def image_block(messages) -> dict:
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue

        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                return block

    raise AssertionError("no image block was sent")


def test_unconfigured_client_says_so():
    assert VisionClient(api_key="").available is False

    with pytest.raises(VisionUnavailable) as error:
        VisionClient(api_key="").recognise_item(JPEG)

    assert "VISION_API_KEY" in str(error.value)


def test_client_is_configured_by_the_environment(monkeypatch):
    monkeypatch.setenv("VISION_API_KEY", "key")
    reset_vision_client()

    assert get_vision_client().available is True

    reset_vision_client()

    monkeypatch.delenv("VISION_API_KEY")

    assert get_vision_client().available is False

    reset_vision_client()


def test_text_model_key_is_not_reused_for_photos(monkeypatch):
    # DeepSeek cannot see images, so its key must not make the client look ready.
    monkeypatch.setenv("API_KEY", "deepseek-key")
    monkeypatch.delenv("VISION_API_KEY", raising=False)

    assert VisionClient().available is False


def test_recognise_item_sends_system_prompt_and_image():
    captured: list[list] = []

    class CapturingLLM:
        def __init__(self, value):
            self.value = value

        def with_structured_output(self, schema):
            class Wrapper:
                def invoke(self, messages):
                    captured.append(messages)

                    return ItemRecognition(name="Свитер кремовый", category="top")

            return Wrapper()

    vision = VisionClient(llm=CapturingLLM(None))
    result = vision.recognise_item(jpeg_of((600, 800)))

    assert result.name == "Свитер кремовый"

    messages = captured[0]

    assert isinstance(messages[0], SystemMessage)
    assert "garment" in messages[0].content
    assert image_block(messages) is not None


def test_item_prompt_forbids_guessing_colour_and_material():
    from fashion_agent.vision import SYSTEM_PROMPTS

    prompt = SYSTEM_PROMPTS["item"].lower()

    assert "most careful" in prompt
    assert "unknown" in prompt
    assert "material must be something the photo supports" in prompt


def test_reference_prompt_refuses_to_invent():
    from fashion_agent.vision import SYSTEM_PROMPTS

    assert "already told you" in SYSTEM_PROMPTS["reference"]
    assert "not a reason" in SYSTEM_PROMPTS["reference"]


def test_critique_prompt_demands_concrete_changes():
    from fashion_agent.vision import SYSTEM_PROMPTS

    assert "concrete" in SYSTEM_PROMPTS["critique"]
    assert "Never suggest a change only to sound thorough" in SYSTEM_PROMPTS["critique"]


def test_read_reference_passes_the_verdict():
    vision = client(ReferenceReading(liked=True, attributes=["color:black"]))

    reading = vision.read_reference(jpeg_of((400, 400)), liked=True)

    assert reading.liked is True


def test_read_reference_tells_the_model_the_client_dislikes_it():
    vision = client(ReferenceReading(liked=False))

    reading = vision.read_reference(jpeg_of((400, 400)), liked=False)

    assert reading.liked is False


def test_critique_passes_occasion_profile_and_wardrobe():
    vision = client(critique())

    vision.critique_look(
        jpeg_of((500, 700)),
        occasion="работа",
        request_note="хочу строго",
        profile_lines=["тип фигуры: груша"],
        wardrobe=[{"id": "abc", "name": "чёрные лоферы"}],
    )

    assert vision.available


def test_critique_score_helpers():
    rated = critique(
        occasion_fit=8,
        cohesion=9,
        colour_harmony=7,
        proportions=6,
        silhouette=10,
    )

    assert rated.total == 40
    assert rated.mean == 8.0


def test_attributes_are_normalised_and_deduped():
    reading = ItemRecognition(
        name="Платье",
        attributes=[" Color:Cream ", "color:cream", "not-an-attribute", "weird:value"],
    )

    assert reading.attributes == ["color:cream"]


def test_color_attributes_only_returns_colors():
    reading = ItemRecognition(
        name="Платье",
        attributes=["color:cream", "fit:oversized", "material:knit"],
    )

    assert reading.color_attributes == ["color:cream"]


def test_unknown_fields_are_kept_rather_than_filled():
    reading = ItemRecognition(name="Платье", unknown=["состав", "бренд"])

    assert reading.unknown == ["состав", "бренд"]
    assert reading.brand is None


def test_look_change_defaults_to_a_replacement():
    change = LookChange(target="обувь", reason="каблук коротковат")

    assert change.action == "replace"
    assert change.wardrobe_item_id is None


def test_look_critique_requires_every_axis():
    with pytest.raises(ValidationError):
        LookCritique(summary="норм")


def test_look_critique_defaults():
    rated = critique()

    assert rated.changes == []
    assert rated.visible_limits == []


def test_provider_errors_become_vision_call_failed():
    class Boom:
        def with_structured_output(self, schema):
            raise RuntimeError("429 rate limit")

    vision = VisionClient(llm=Boom())

    with pytest.raises(VisionCallFailed) as error:
        vision.recognise_item(jpeg_of((100, 100)))

    assert "429" in str(error.value)


def test_image_is_downscaled_to_a_sane_size():
    normalized = normalize_image(jpeg_of((4000, 3000)))

    with Image.open(io.BytesIO(normalized)) as image:
        assert max(image.size) <= 1024


def test_small_image_keeps_its_size():
    normalized = normalize_image(jpeg_of((300, 200)))

    with Image.open(io.BytesIO(normalized)) as image:
        assert image.size == (300, 200)


def test_image_is_converted_to_rgb_jpeg():
    buffer = io.BytesIO()
    Image.new("RGBA", (100, 100), (1, 2, 3, 4)).save(buffer, "PNG")

    normalized = normalize_image(buffer.getvalue())

    with Image.open(io.BytesIO(normalized)) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"


def test_image_is_sent_base64_in_a_data_url():
    captured: list[list] = []

    class CapturingLLM(StubLLM):
        def with_structured_output(self, schema):
            class Wrapper:
                def invoke(self, messages):
                    captured.append(messages)

                    return ItemRecognition(name="Свитер")

            return Wrapper()

    vision = VisionClient(llm=CapturingLLM(ItemRecognition(name="Свитер")))
    vision.recognise_item(jpeg_of((200, 200)))

    url = image_block(captured[0])["image_url"]["url"]

    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1])[:3] == b"\xff\xd8\xff"


def test_category_hint_reaches_the_model():
    captured: list[list] = []

    class CapturingLLM:
        def with_structured_output(self, schema):
            class Wrapper:
                def invoke(self, messages):
                    captured.append(messages)

                    return ItemRecognition(name="Платье")

            return Wrapper()

    vision = VisionClient(llm=CapturingLLM())
    vision.recognise_item(jpeg_of((200, 200)), category_hint="платье", context="новое")

    text = captured[0][-1].content[0]["text"]

    assert "платье" in text
    assert "новое" in text
