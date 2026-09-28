from src.fashion_agent import web_collage


def test_build_collage_html_embeds_cutouts():
    html = web_collage.build_collage_html(
        outfit={
            "total_price": 43800,
            "currency": "RUB",
        },
        cutouts=[
            {
                "title": "Black dress",
                "price": 30000,
                "currency": "RUB",
                "source": "Shop A",
                "image_data_url": "data:image/png;base64,aaa",
            },
            {
                "title": "Mary Janes",
                "price": 13800,
                "currency": "RUB",
                "source": "Shop B",
                "image_data_url": "data:image/png;base64,bbb",
            },
        ],
    )

    assert "Black dress" in html
    assert "Mary Janes" in html
    assert "data:image/png;base64,aaa" in html
    assert "43 800 ₽" in html


def test_build_outfit_collage_data_url_returns_none_without_images():
    result = web_collage.build_outfit_collage_data_url(
        {
            "items": [
                {
                    "title": "Dress",
                }
            ]
        }
    )

    assert result is None


def test_remove_background_falls_back_without_onnxruntime(monkeypatch):
    class BrokenRembg:
        @staticmethod
        def remove(
            image_bytes,
        ):
            raise RuntimeError("No onnxruntime backend found.")

    monkeypatch.setattr(
        web_collage,
        "normalize_png",
        lambda image_bytes: b"normalized",
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "rembg",
        BrokenRembg(),
    )

    result = web_collage.remove_background(
        b"raw-image",
    )

    assert result == b"normalized"


def test_build_outfit_collage_data_url_runs_pipeline(monkeypatch):
    called = {}

    def fake_render_html_to_png(
        html_string,
    ):
        called["html"] = html_string
        return b"png"

    monkeypatch.setattr(
        web_collage,
        "download_image_bytes",
        lambda url: b"raw-image",
    )
    monkeypatch.setattr(
        web_collage,
        "remove_background",
        lambda image_bytes: b"cutout-image",
    )
    monkeypatch.setattr(
        web_collage,
        "render_html_to_png",
        fake_render_html_to_png,
    )
    monkeypatch.setattr(
        web_collage,
        "image_bytes_to_data_url",
        lambda image_bytes, mime_type="image/png": (
            f"data:{mime_type};base64,{image_bytes.decode('latin1')}"
        ),
    )

    result = web_collage.build_outfit_collage_data_url(
        {
            "total_price": 12000,
            "currency": "RUB",
            "items": [
                {
                    "title": "Black dress",
                    "price": 12000,
                    "currency": "RUB",
                    "source": "Shop",
                    "image_url": "https://example.com/1.jpg",
                }
            ],
        }
    )

    assert result == "data:image/png;base64,png"
    assert "Black dress" in called["html"]
