import base64
import functools
import importlib.metadata
import inspect
import itertools
import json
import logging
import ssl
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import certifi
import gradio as gr
import httpx

from docling.datamodel.base_models import FormatToExtensions
from docling.datamodel.pipeline_options import (
    PdfBackend,
    ProcessingPipeline,
    TableFormerMode,
    TableStructureOptions,
)

from docling_serve.helper_functions import _to_list_of_strings
from docling_serve.settings import docling_serve_settings, uvicorn_settings

logger = logging.getLogger(__name__)

try:
    docling_version = importlib.metadata.version("docling")
except (importlib.metadata.PackageNotFoundError, AttributeError):
    try:
        docling_version = importlib.metadata.version("docling-slim")
    except (importlib.metadata.PackageNotFoundError, AttributeError):
        docling_version = "unknown"

############################
# Path of static artifacts #
############################

logo_path = "https://raw.githubusercontent.com/docling-project/docling/refs/heads/main/docs/assets/logo.svg"
js_components_url = "https://unpkg.com/@docling/docling-components@0.0.7"
if (
    docling_serve_settings.static_path is not None
    and docling_serve_settings.static_path.is_dir()
):
    logo_path = str(docling_serve_settings.static_path / "logo.svg")
    js_components_url = "/static/docling-components.js"


##############################
# Head JS for web components #
##############################
head = f"""
    <script src="{js_components_url}" type="module"></script>
"""

#################
# CSS and theme #
#################

css = """
#logo {
    border-style: none;
    background: none;
    box-shadow: none;
    min-width: 80px;
}
#dark_mode_column {
    display: flex;
    align-content: flex-end;
}
#title {
    text-align: left;
    display:block;
    height: auto;
    padding-top: 5px;
    line-height: 0;
}
.title-text h1 > p, .title-text p {
    margin-top: 0px !important;
    margin-bottom: 2px !important;
}
#custom-container {
    border: 0.909091px solid;
    padding: 10px;
    border-radius: 4px;
}
#custom-container h4 {
    font-size: 14px;
}
#file_input_zone {
    height: 140px;
}

docling-img {
    gap: 1rem;
}

docling-img::part(page) {
    box-shadow: 0 0.5rem 1rem 0 rgba(0, 0, 0, 0.2);
}

#dclroot {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.dcl-legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem 0.9rem;
    font-size: 0.8rem;
}

.dcl-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    white-space: nowrap;
}

.dcl-swatch {
    width: 0.85rem;
    height: 0.85rem;
    border-radius: 2px;
    border: 1px solid rgba(0, 0, 0, 0.25);
}

.dcl-toggle-label {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    margin-right: 0.5rem;
    font-weight: 600;
    cursor: pointer;
}

.dcl-notice,
.dcl-status:not(:empty) {
    padding: 0.75rem 1rem;
    border: 1px solid rgba(128, 128, 128, 0.4);
    border-radius: 4px;
    font-size: 0.9rem;
    line-height: 1.4;
}

.dcl-notice code,
.dcl-status code {
    font-size: 0.85em;
}

.dcl-boot {
    display: none;
}
"""

theme = gr.themes.Default(
    text_size="md",
    spacing_size="md",
    font=[
        gr.themes.GoogleFont("Red Hat Display"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("Red Hat Mono"),
        "ui-monospace",
        "Consolas",
        "monospace",
    ],
)

#############
# Variables #
#############

gradio_output_dir = None  # Will be set by FastAPI when mounted
file_output_path = None  # Will be set when a new file is generated

#############
# Functions #
#############


def get_api_endpoint() -> str:
    protocol = "http"
    if uvicorn_settings.ssl_keyfile is not None:
        protocol = "https"
    return f"{protocol}://{docling_serve_settings.api_host}:{uvicorn_settings.port}"


def get_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context(cafile=certifi.where())
    kube_sa_ca_cert_path = Path(
        "/run/secrets/kubernetes.io/serviceaccount/service-ca.crt"
    )
    if (
        uvicorn_settings.ssl_keyfile is not None
        and ".svc." in docling_serve_settings.api_host
        and kube_sa_ca_cert_path.exists()
    ):
        ctx.load_verify_locations(cafile=kube_sa_ca_cert_path)
    return ctx


def health_check():
    response = httpx.get(f"{get_api_endpoint()}/health")
    if response.status_code == 200:
        return "Healthy"
    return "Unhealthy"


def set_options_visibility(x):
    return gr.Accordion("Options", open=x)


def set_outputs_visibility_direct(x, y):
    content = gr.Row(visible=x)
    file = gr.Row(visible=y)
    return content, file


def set_task_id_visibility(x):
    task_id_row = gr.Row(visible=x)
    return task_id_row


def set_outputs_visibility_process(x):
    content = gr.Row(visible=not x)
    file = gr.Row(visible=x)
    return content, file


def set_download_button_label(label_text: gr.State):
    return gr.DownloadButton(label=str(label_text), scale=1)


def clear_outputs():
    task_id_rendered = ""
    markdown_content = ""
    json_content = ""
    json_rendered_content = ""
    html_content = ""
    text_content = ""
    doctags_content = ""

    return (
        task_id_rendered,
        markdown_content,
        markdown_content,
        json_content,
        json_rendered_content,
        html_content,
        html_content,
        text_content,
        doctags_content,
    )


def clear_url_input():
    return ""


def clear_file_input():
    return None


def auto_set_return_as_file(
    url_input_value: str,
    file_input_value: Optional[list[str]],
    image_export_mode_value: str,
):
    # If more than one input source is provided, return as file
    if (
        (len(url_input_value.split(",")) > 1)
        or (file_input_value and len(file_input_value) > 1)
        or (image_export_mode_value == "referenced")
    ):
        return True
    else:
        return False


def change_ocr_lang(ocr_engine):
    if ocr_engine == "easyocr":
        return gr.update(visible=True, value="de,en,fr,es")
    elif ocr_engine == "tesseract_cli":
        return gr.update(visible=True, value="deu,eng,fra,spa")
    elif ocr_engine == "tesseract":
        return gr.update(visible=True, value="deu,eng,fra,spa")
    elif ocr_engine == "rapidocr":
        return gr.update(visible=True, value="de,en,fr,es,ch")
    elif ocr_engine == "ocrmac":
        return gr.update(visible=True, value="de-DE,en-US,fr-FR,es-ES")

    return gr.update(visible=False, value="")


def wait_task_finish(auth: str, task_id: str, return_as_file: bool):
    conversion_sucess = False
    task_finished = False
    task_status = ""

    headers = {}
    if docling_serve_settings.api_key:
        headers["X-Api-Key"] = str(auth)

    ssl_ctx = get_ssl_context()
    while not task_finished:
        try:
            response = httpx.get(
                f"{get_api_endpoint()}/v1/status/poll/{task_id}?wait=5",
                headers=headers,
                verify=ssl_ctx,
                timeout=15,
            )

            # Check response status code first
            if response.status_code == 404:
                logger.warning(
                    f"Task {task_id} not found in status poll, it may have completed already"
                )
                time.sleep(2)  # Wait for result to be ready
                conversion_sucess = True
                task_finished = True
                break

            response.raise_for_status()

            # Safely access task_status
            response_data = response.json()
            if "task_status" not in response_data:
                logger.error(f"Missing task_status in response: {response_data}")
                raise RuntimeError("Missing task_status in response")

            task_status = response_data["task_status"]
            if task_status == "success":
                conversion_sucess = True
                task_finished = True

            if task_status in ("failure", "revoked"):
                conversion_sucess = False
                task_finished = True
                raise RuntimeError(f"Task failed with status {task_status!r}")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Error processing file(s): {e}")
            conversion_sucess = False
            task_finished = True
            raise gr.Error(f"Error processing file(s): {e}", print_exception=False)

    # Retry logic for result retrieval
    if conversion_sucess:
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = httpx.get(
                    f"{get_api_endpoint()}/v1/result/{task_id}",
                    headers=headers,
                    timeout=15,
                    verify=ssl_ctx,
                )

                if response.status_code == 404:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 2**retry_count  # Exponential backoff: 2, 4, 8s
                        logger.warning(
                            f"Result not ready yet, retrying in {wait_time}s "
                            f"(attempt {retry_count}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(
                            f"Result not available after {max_retries} retries"
                        )
                        raise RuntimeError(
                            f"Result not available after {max_retries} retries"
                        )

                response.raise_for_status()
                output = response_to_output(response, return_as_file)
                return output
            except Exception as e:
                if retry_count >= max_retries - 1:
                    logger.error(f"Error getting task result: {e}")
                    raise gr.Error(
                        f"Error getting task result: {e}", print_exception=False
                    )
                # For non-404 errors on early retries, continue retrying
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2**retry_count
                    logger.warning(
                        f"Error getting result, retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)

    raise gr.Error(
        f"Error getting task result, conversion finished with status: {task_status}"
    )


def process_url(
    auth,
    input_sources,
    to_formats,
    image_export_mode,
    include_page_images,
    pipeline,
    layout_engine,
    ocr,
    force_ocr,
    ocr_engine,
    ocr_lang,
    pdf_backend,
    table_mode,
    heading_hierarchy,
    abort_on_error,
    return_as_file,
    do_code_enrichment,
    do_formula_enrichment,
    do_picture_classification,
    do_picture_description,
):
    target = {"kind": "zip" if return_as_file else "inbody"}
    parameters = {
        "sources": [
            {"kind": "http", "url": source} for source in input_sources.split(",")
        ],
        "options": {
            "to_formats": to_formats,
            "image_export_mode": image_export_mode,
            "include_page_images": include_page_images,
            "pipeline": pipeline,
            "do_ocr": ocr,
            "force_ocr": force_ocr,
            "ocr_preset": ocr_engine,
            "ocr_lang": _to_list_of_strings(ocr_lang),
            "pdf_backend": pdf_backend,
            "table_mode": table_mode,
            "do_pdf_heading_hierarchy": heading_hierarchy,
            "abort_on_error": abort_on_error,
            "do_code_enrichment": do_code_enrichment,
            "do_formula_enrichment": do_formula_enrichment,
            "do_picture_classification": do_picture_classification,
            "do_picture_description": do_picture_description,
        },
        "target": target,
    }

    if layout_engine != "default":
        parameters["options"]["layout_custom_config"] = {"kind": layout_engine}

    if (
        not parameters["sources"]
        or len(parameters["sources"]) == 0
        or parameters["sources"][0]["url"] == ""
    ):
        logger.error("No input sources provided.")
        raise gr.Error("No input sources provided.", print_exception=False)

    headers = {}
    if docling_serve_settings.api_key:
        headers["X-Api-Key"] = str(auth)

    try:
        ssl_ctx = get_ssl_context()
        response = httpx.post(
            f"{get_api_endpoint()}/v1/convert/source/async",
            json=parameters,
            headers=headers,
            verify=ssl_ctx,
            timeout=60,
        )
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        raise gr.Error(f"Error processing URL: {e}", print_exception=False)
    if response.status_code != 200:
        data = response.json()
        error_message = data.get("detail", "An unknown error occurred.")
        logger.error(f"Error processing file: {error_message}")
        raise gr.Error(f"Error processing file: {error_message}", print_exception=False)

    task_id_rendered = response.json()["task_id"]
    return task_id_rendered


def file_to_base64(file):
    with open(file.name, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode("utf-8")
    return encoded_string


def process_file(
    auth,
    files,
    to_formats,
    image_export_mode,
    include_page_images,
    pipeline,
    layout_engine,
    ocr,
    force_ocr,
    ocr_engine,
    ocr_lang,
    pdf_backend,
    table_mode,
    heading_hierarchy,
    abort_on_error,
    return_as_file,
    do_code_enrichment,
    do_formula_enrichment,
    do_picture_classification,
    do_picture_description,
):
    if not files or len(files) == 0:
        logger.error("No files provided.")
        raise gr.Error("No files provided.", print_exception=False)
    files_data = [
        {"kind": "file", "base64_string": file_to_base64(file), "filename": file.name}
        for file in files
    ]
    target = {"kind": "zip" if return_as_file else "inbody"}

    parameters = {
        "sources": files_data,
        "options": {
            "to_formats": to_formats,
            "image_export_mode": image_export_mode,
            "include_page_images": include_page_images,
            "pipeline": pipeline,
            "do_ocr": ocr,
            "force_ocr": force_ocr,
            "ocr_preset": ocr_engine,
            "ocr_lang": _to_list_of_strings(ocr_lang),
            "pdf_backend": pdf_backend,
            "table_mode": table_mode,
            "do_pdf_heading_hierarchy": heading_hierarchy,
            "abort_on_error": abort_on_error,
            "return_as_file": return_as_file,
            "do_code_enrichment": do_code_enrichment,
            "do_formula_enrichment": do_formula_enrichment,
            "do_picture_classification": do_picture_classification,
            "do_picture_description": do_picture_description,
        },
        "target": target,
    }

    if layout_engine != "default":
        parameters["options"]["layout_custom_config"] = {"kind": layout_engine}

    headers = {}
    if docling_serve_settings.api_key:
        headers["X-Api-Key"] = str(auth)

    try:
        ssl_ctx = get_ssl_context()
        response = httpx.post(
            f"{get_api_endpoint()}/v1/convert/source/async",
            json=parameters,
            headers=headers,
            verify=ssl_ctx,
            timeout=60,
        )
    except Exception as e:
        logger.error(f"Error processing file(s): {e}")
        raise gr.Error(f"Error processing file(s): {e}", print_exception=False)
    if response.status_code != 200:
        data = response.json()
        error_message = data.get("detail", "An unknown error occurred.")
        logger.error(f"Error processing file: {error_message}")
        raise gr.Error(f"Error processing file: {error_message}", print_exception=False)

    task_id_rendered = response.json()["task_id"]
    return task_id_rendered


#############################
# Docling-Rendered bboxes   #
#############################

# Stroke/fill colour per DocItemLabel, used for the bounding boxes drawn on top
# of the page images in the "Docling-Rendered" tab.
BBOX_COLORS = {
    "title": "#d7263d",
    "section_header": "#f46036",
    "text": "#2e86ab",
    "paragraph": "#2e86ab",
    "list_item": "#6a4c93",
    "table": "#1b998b",
    "picture": "#c5299b",
    "chart": "#06d6a0",
    "caption": "#ff9f1c",
    "formula": "#7d5ba6",
    "code": "#3f8efc",
    "page_header": "#8d99ae",
    "page_footer": "#8d99ae",
    "footnote": "#a68a64",
    "document_index": "#0f7173",
    "reference": "#b56576",
    "checkbox_selected": "#4c956c",
    "checkbox_unselected": "#4c956c",
    "form": "#8338ec",
    "key_value_region": "#fb5607",
}
BBOX_DEFAULT_COLOR = "#6c757d"

# Item collections of a DoclingDocument whose entries carry a label and prov.
_LABELLED_COLLECTIONS = (
    "texts",
    "tables",
    "pictures",
    "form_items",
    "key_value_items",
)

# Runs once the docling web components are registered: parses the embedded
# document, colours every bounding box by its label and wires the toggle.
_BOOTSTRAP_JS = """
(function () {
  var root = document.getElementById('dclroot');
  if (!root) return;
  var img = root.querySelector('docling-img');
  var raw = root.querySelector('script[type="application/json"]');
  var status = root.querySelector('.dcl-status');
  var doc;
  try {
    doc = JSON.parse(raw.textContent);
  } catch (err) {
    status.textContent = 'Could not parse the Docling JSON: ' + err.message;
    return;
  }

  var colors = __COLORS__;
  var fallback = '__DEFAULT_COLOR__';
  var visible = true;

  function itemStyle(page, item) {
    if (!visible) return 'visibility: hidden';
    var color = (item && colors[item.label]) || fallback;
    return 'stroke: ' + color + '; fill: ' + color +
           '; stroke-width: 1.5px; stroke-dasharray: none';
  }

  function applyStyle() {
    // A fresh function reference is what makes Lit re-render the boxes.
    img.itemStyle = function (page, item) { return itemStyle(page, item); };
  }

  var toggle = root.querySelector('.dcl-toggle');
  if (toggle) {
    toggle.addEventListener('change', function () {
      visible = toggle.checked;
      applyStyle();
    });
  }

  var timeout = setTimeout(function () {
    status.innerHTML = 'The Docling web components could not be loaded from ' +
      '<code>__JS_URL__</code>. Check that this page can reach it.';
  }, 15000);

  customElements.whenDefined('docling-img').then(function () {
    clearTimeout(timeout);
    status.textContent = '';
    img.itemPart = function (page, item) {
      return item && item.label ? String(item.label) : '';
    };
    applyStyle();
    img.src = doc;
  });
})();
"""


def document_bbox_labels(document: dict) -> list[str]:
    """Labels of every item that carries provenance, in first-seen order."""
    labels: list[str] = []
    for collection in _LABELLED_COLLECTIONS:
        for item in document.get(collection) or []:
            if not isinstance(item, dict) or not item.get("prov"):
                continue
            label = item.get("label")
            if label and label not in labels:
                labels.append(str(label))
    return labels


def document_has_page_images(document: dict) -> bool:
    pages = document.get("pages") or {}
    if not isinstance(pages, dict):
        return False
    return any(isinstance(page, dict) and page.get("image") for page in pages.values())


def build_json_rendered(document: Optional[dict], json_text: str) -> str:
    """HTML for the Docling-Rendered tab: page images plus labelled bboxes."""
    if not isinstance(document, dict) or not document:
        return (
            '<div class="dcl-notice">No Docling JSON in the response. '
            "Select <b>Docling (JSON)</b> under <b>To Formats</b> to use this view."
            "</div>"
        )
    if not document_has_page_images(document):
        return (
            '<div class="dcl-notice">This document carries no page images, so there '
            "is nothing to draw the bounding boxes on. Enable <b>Include page "
            "images</b> in <b>Options</b> and convert again.</div>"
        )

    labels = document_bbox_labels(document)
    legend = "".join(
        '<span class="dcl-legend-item">'
        f'<span class="dcl-swatch" style="background: {BBOX_COLORS.get(label, BBOX_DEFAULT_COLOR)}"></span>'
        f"{label}</span>"
        for label in labels
    )
    bootstrap = (
        _BOOTSTRAP_JS.replace("__COLORS__", json.dumps(BBOX_COLORS))
        .replace("__DEFAULT_COLOR__", BBOX_DEFAULT_COLOR)
        .replace("__JS_URL__", js_components_url)
    )
    # Escaped so document text can never terminate the <script> element early.
    embedded_json = json_text.replace("</", "<\\/")

    return f"""
        <div id="dclroot">
          <div class="dcl-legend">
            <label class="dcl-toggle-label">
              <input class="dcl-toggle" type="checkbox" checked /> Bounding boxes
            </label>
            {legend}
          </div>
          <div class="dcl-status">Loading the Docling viewer...</div>
          <docling-img pagenumbers><docling-tooltip></docling-tooltip></docling-img>
          <script type="application/json">{embedded_json}</script>
          <script id="dclboot" type="text/plain">{bootstrap}</script>
        </div>
        <img class="dcl-boot" alt="" src="data:," onerror="var b=document.getElementById('dclboot');var s=document.createElement('script');s.textContent=b.textContent;document.body.appendChild(s);s.remove();this.remove();" />
        """


def response_to_output(response, return_as_file):
    markdown_content = ""
    json_content = ""
    json_rendered_content = ""
    html_content = ""
    text_content = ""
    doctags_content = ""
    download_button = gr.DownloadButton(visible=False, label="Download Output", scale=1)
    if return_as_file:
        filename = (
            response.headers.get("Content-Disposition").split("filename=")[1].strip('"')
        )
        tmp_output_dir = Path(tempfile.mkdtemp(dir=gradio_output_dir, prefix="ui_"))
        file_output_path = f"{tmp_output_dir}/{filename}"
        # logger.info(f"Saving file to: {file_output_path}")
        with open(file_output_path, "wb") as f:
            f.write(response.content)
        download_button = gr.DownloadButton(
            visible=True, label=f"Download {filename}", scale=1, value=file_output_path
        )
    else:
        full_content = response.json()
        markdown_content = full_content.get("document").get("md_content")
        document_json = full_content.get("document").get("json_content")
        json_content = json.dumps(document_json, indent=2)
        json_rendered_content = build_json_rendered(document_json, json_content)
        html_content = full_content.get("document").get("html_content")
        text_content = full_content.get("document").get("text_content")
        doctags_content = full_content.get("document").get("doctags_content")
    return (
        markdown_content,
        markdown_content,
        json_content,
        json_rendered_content,
        html_content,
        html_content,
        text_content,
        doctags_content,
        download_button,
    )


############
# UI Setup #
############

with gr.Blocks(
    head=head,
    css=css,
    theme=theme,
    title="Docling Serve",
    delete_cache=(3600, 36000),  # Delete all files older than 10 hour every hour
) as ui:
    # Constants stored in states to be able to pass them as inputs to functions
    processing_text = gr.State("Processing your document(s), please wait...")
    true_bool = gr.State(True)
    false_bool = gr.State(False)

    # Banner
    with gr.Row(elem_id="check_health"):
        # Logo
        with gr.Column(scale=1, min_width=90):
            try:
                gr.Image(
                    logo_path,
                    height=80,
                    width=80,
                    buttons=[],
                    show_label=False,
                    container=False,
                    elem_id="logo",
                    scale=0,
                )
            except Exception:
                logger.warning("Logo not found.")

        # Title
        with gr.Column(scale=1, min_width=200):
            gr.Markdown(
                f"# Docling Serve \n(docling version: {docling_version})",
                elem_id="title",
                elem_classes=["title-text"],
            )
        # Dark mode button
        with gr.Column(scale=16, elem_id="dark_mode_column"):
            dark_mode_btn = gr.Button("Dark/Light Mode", scale=0)
            dark_mode_btn.click(
                None,
                None,
                None,
                js="""() => {
                    if (document.querySelectorAll('.dark').length) {
                        document.querySelectorAll('.dark').forEach(
                        el => el.classList.remove('dark')
                        );
                    } else {
                        document.querySelector('body').classList.add('dark');
                    }
                }""",
                api_visibility="undocumented",
            )

    # URL Processing Tab
    with gr.Tab("Convert URL"):
        with gr.Row():
            with gr.Column(scale=4):
                url_input = gr.Textbox(
                    label="URL Input Source",
                    placeholder="https://arxiv.org/pdf/2501.17887",
                )
            with gr.Column(scale=1):
                url_process_btn = gr.Button("Process URL", scale=1)
                url_reset_btn = gr.Button("Reset", scale=1)

    # File Processing Tab
    with gr.Tab("Convert File"):
        with gr.Row():
            with gr.Column(scale=4):
                raw_exts = itertools.chain.from_iterable(FormatToExtensions.values())
                file_input = gr.File(
                    elem_id="file_input_zone",
                    label="Upload File",
                    file_types=[
                        f".{v.lower()}"
                        for v in raw_exts  # lowercase
                    ]
                    + [
                        f".{v.upper()}"
                        for v in raw_exts  # uppercase
                    ],
                    file_count="multiple",
                    scale=4,
                )
            with gr.Column(scale=1):
                file_process_btn = gr.Button("Process File", scale=1)
                file_reset_btn = gr.Button("Reset", scale=1)

    # Auth
    with gr.Row(visible=bool(docling_serve_settings.api_key)):
        with gr.Column():
            auth = gr.Textbox(
                label="Authentication",
                placeholder="API Key",
                type="password",
            )

    # Options
    with gr.Accordion("Options") as options:
        with gr.Row():
            with gr.Column(scale=1):
                to_formats = gr.CheckboxGroup(
                    [
                        ("Docling (JSON)", "json"),
                        ("Markdown", "md"),
                        ("HTML", "html"),
                        ("Plain Text", "text"),
                        ("Doc Tags", "doctags"),
                    ],
                    label="To Formats",
                    value=["json", "md"],
                )
            with gr.Column(scale=1):
                image_export_mode = gr.Radio(
                    [
                        ("Embedded", "embedded"),
                        ("Placeholder", "placeholder"),
                        ("Referenced", "referenced"),
                    ],
                    label="Image Export Mode",
                    value="embedded",
                )
                include_page_images = gr.Checkbox(
                    label="Include page images",
                    info=(
                        "Embed a full-page image per page. Required by the "
                        "Docling-Rendered tab, which draws the bounding boxes on "
                        "top of them. Noticeably increases the response size."
                    ),
                    value=False,
                )

        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                pipeline = gr.Radio(
                    [(v.value.capitalize(), v.value) for v in ProcessingPipeline],
                    label="Pipeline type",
                    value=ProcessingPipeline.STANDARD.value,
                )
            with gr.Column(scale=1, min_width=200):
                layout_engine = gr.Radio(
                    [
                        ("Default", "default"),
                        ("PP-DocLayout-V3", "ppdoclayout-v3"),
                    ],
                    label="Layout Engine",
                    value="default",
                )
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                ocr = gr.Checkbox(label="Enable OCR", value=True)
                force_ocr = gr.Checkbox(label="Force OCR", value=False)
            with gr.Column(scale=1):
                engines_list = [
                    ("Auto", "auto"),
                    ("EasyOCR", "easyocr"),
                    ("Tesseract", "tesseract"),
                    ("RapidOCR", "rapidocr"),
                    ("GLM-OCR (Remote)", "glm-ocr-remote"),
                ]
                if sys.platform == "darwin":
                    engines_list.append(("OCRMac", "ocrmac"))

                ocr_engine = gr.Radio(
                    engines_list,
                    label="OCR Engine",
                    value="auto",
                )
            with gr.Column(scale=1, min_width=200):
                ocr_lang = gr.Textbox(
                    label="OCR Language (beware of the format)",
                    value="de,en,fr,es",
                    visible=False,
                )
            ocr_engine.change(change_ocr_lang, inputs=[ocr_engine], outputs=[ocr_lang])
        with gr.Row():
            with gr.Column(scale=4):
                pdf_backend = gr.Radio(
                    [v.value for v in (PdfBackend.DOCLING_PARSE, PdfBackend.PYPDFIUM2)],
                    label="PDF Backend",
                    value=PdfBackend.DOCLING_PARSE.value,
                )
            with gr.Column(scale=2):
                table_mode = gr.Radio(
                    [(v.value.capitalize(), v.value) for v in TableFormerMode],
                    label="Table Mode",
                    value=TableStructureOptions().mode.value,
                )
            with gr.Column(scale=2):
                heading_hierarchy = gr.Checkbox(
                    label="Infer heading levels",
                    info=(
                        "Assign section-header levels from the PDF bookmarks, "
                        "numbering and font style instead of leaving every heading "
                        "at level 1."
                    ),
                    value=False,
                )
            with gr.Column(scale=1):
                abort_on_error = gr.Checkbox(label="Abort on Error", value=False)
                return_as_file = gr.Checkbox(label="Return as File", value=False)
        with gr.Row():
            with gr.Column():
                do_code_enrichment = gr.Checkbox(
                    label="Enable code enrichment", value=False
                )
                do_formula_enrichment = gr.Checkbox(
                    label="Enable formula enrichment", value=False
                )
            with gr.Column():
                do_picture_classification = gr.Checkbox(
                    label="Enable picture classification", value=False
                )
                do_picture_description = gr.Checkbox(
                    label="Enable picture description", value=False
                )

    # Task id output
    with gr.Row(visible=False) as task_id_output:
        task_id_rendered = gr.Textbox(label="Task id", interactive=False)

    # Document output
    with gr.Row(visible=False) as content_output:
        with gr.Tab("Docling (JSON)"):
            output_json = gr.Code(language="json", wrap_lines=True, show_label=False)
        with gr.Tab("Docling-Rendered"):
            output_json_rendered = gr.HTML(label="Response")
        with gr.Tab("Markdown"):
            output_markdown = gr.Code(
                language="markdown", wrap_lines=True, show_label=False
            )
        with gr.Tab("Markdown-Rendered"):
            output_markdown_rendered = gr.Markdown(label="Response")
        with gr.Tab("HTML"):
            output_html = gr.Code(language="html", wrap_lines=True, show_label=False)
        with gr.Tab("HTML-Rendered"):
            output_html_rendered = gr.HTML(label="Response")
        with gr.Tab("Text"):
            output_text = gr.Code(wrap_lines=True, show_label=False)
        with gr.Tab("DocTags"):
            output_doctags = gr.Code(wrap_lines=True, show_label=False)

    # File download output
    with gr.Row(visible=False) as file_output:
        download_file_btn = gr.DownloadButton(label="Placeholder", scale=1)

    ##############
    # UI Actions #
    ##############

    # Handle Return as File
    url_input.change(
        auto_set_return_as_file,
        inputs=[url_input, file_input, image_export_mode],
        outputs=[return_as_file],
    )
    file_input.change(
        auto_set_return_as_file,
        inputs=[url_input, file_input, image_export_mode],
        outputs=[return_as_file],
    )
    image_export_mode.change(
        auto_set_return_as_file,
        inputs=[url_input, file_input, image_export_mode],
        outputs=[return_as_file],
    )

    # URL processing
    url_process_btn.click(
        set_options_visibility, inputs=[false_bool], outputs=[options]
    ).then(
        set_download_button_label, inputs=[processing_text], outputs=[download_file_btn]
    ).then(
        clear_outputs,
        inputs=None,
        outputs=[
            task_id_rendered,
            output_markdown,
            output_markdown_rendered,
            output_json,
            output_json_rendered,
            output_html,
            output_html_rendered,
            output_text,
            output_doctags,
        ],
    ).then(
        set_task_id_visibility,
        inputs=[true_bool],
        outputs=[task_id_output],
    ).then(
        process_url,
        inputs=[
            auth,
            url_input,
            to_formats,
            image_export_mode,
            include_page_images,
            pipeline,
            layout_engine,
            ocr,
            force_ocr,
            ocr_engine,
            ocr_lang,
            pdf_backend,
            table_mode,
            heading_hierarchy,
            abort_on_error,
            return_as_file,
            do_code_enrichment,
            do_formula_enrichment,
            do_picture_classification,
            do_picture_description,
        ],
        outputs=[
            task_id_rendered,
        ],
    ).then(
        set_outputs_visibility_process,
        inputs=[return_as_file],
        outputs=[content_output, file_output],
    ).then(
        wait_task_finish,
        inputs=[auth, task_id_rendered, return_as_file],
        outputs=[
            output_markdown,
            output_markdown_rendered,
            output_json,
            output_json_rendered,
            output_html,
            output_html_rendered,
            output_text,
            output_doctags,
            download_file_btn,
        ],
    )

    url_reset_btn.click(
        clear_outputs,
        inputs=None,
        outputs=[
            output_markdown,
            output_markdown_rendered,
            output_json,
            output_json_rendered,
            output_html,
            output_html_rendered,
            output_text,
            output_doctags,
        ],
    ).then(set_options_visibility, inputs=[true_bool], outputs=[options]).then(
        set_outputs_visibility_direct,
        inputs=[false_bool, false_bool],
        outputs=[content_output, file_output],
    ).then(set_task_id_visibility, inputs=[false_bool], outputs=[task_id_output]).then(
        clear_url_input, inputs=None, outputs=[url_input]
    )

    # File processing
    file_process_btn.click(
        set_options_visibility, inputs=[false_bool], outputs=[options]
    ).then(
        set_download_button_label, inputs=[processing_text], outputs=[download_file_btn]
    ).then(
        clear_outputs,
        inputs=None,
        outputs=[
            task_id_rendered,
            output_markdown,
            output_markdown_rendered,
            output_json,
            output_json_rendered,
            output_html,
            output_html_rendered,
            output_text,
            output_doctags,
        ],
    ).then(
        set_task_id_visibility,
        inputs=[true_bool],
        outputs=[task_id_output],
    ).then(
        process_file,
        inputs=[
            auth,
            file_input,
            to_formats,
            image_export_mode,
            include_page_images,
            pipeline,
            layout_engine,
            ocr,
            force_ocr,
            ocr_engine,
            ocr_lang,
            pdf_backend,
            table_mode,
            heading_hierarchy,
            abort_on_error,
            return_as_file,
            do_code_enrichment,
            do_formula_enrichment,
            do_picture_classification,
            do_picture_description,
        ],
        outputs=[
            task_id_rendered,
        ],
    ).then(
        set_outputs_visibility_process,
        inputs=[return_as_file],
        outputs=[content_output, file_output],
    ).then(
        wait_task_finish,
        inputs=[auth, task_id_rendered, return_as_file],
        outputs=[
            output_markdown,
            output_markdown_rendered,
            output_json,
            output_json_rendered,
            output_html,
            output_html_rendered,
            output_text,
            output_doctags,
            download_file_btn,
        ],
    )

    file_reset_btn.click(
        clear_outputs,
        inputs=None,
        outputs=[
            output_markdown,
            output_markdown_rendered,
            output_json,
            output_json_rendered,
            output_html,
            output_html_rendered,
            output_text,
            output_doctags,
        ],
    ).then(set_options_visibility, inputs=[true_bool], outputs=[options]).then(
        set_outputs_visibility_direct,
        inputs=[false_bool, false_bool],
        outputs=[content_output, file_output],
    ).then(set_task_id_visibility, inputs=[false_bool], outputs=[task_id_output]).then(
        clear_file_input, inputs=None, outputs=[file_input]
    )


####################################
# Gradio 6 head/css/theme fallback #
####################################


def _patch_mount_gradio_app() -> None:
    """Pass ``head``/``css``/``theme`` on to ``mount_gradio_app``.

    Gradio 6 removed those arguments from ``gr.Blocks.__init__`` (they are
    swallowed by ``**kwargs``) and moved them to ``launch()`` and
    ``mount_gradio_app()``, which overwrites them with empty defaults.
    docling-serve mounts this UI without passing them, so without this wrapper
    the docling web components are never loaded and the custom CSS is dropped.
    """
    original = getattr(gr, "mount_gradio_app", None)
    if original is None or getattr(original, "_docling_serve_patched", False):
        return

    try:
        supported = set(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        return

    extras = {
        name: value
        for name, value in (("head", head), ("css", css), ("theme", theme))
        if name in supported
    }
    if not extras:
        return

    @functools.wraps(original)
    def mount_gradio_app(app, blocks, *args, **kwargs):
        if blocks is ui:
            for name, value in extras.items():
                kwargs.setdefault(name, value)
        return original(app, blocks, *args, **kwargs)

    mount_gradio_app._docling_serve_patched = True
    gr.mount_gradio_app = mount_gradio_app


_patch_mount_gradio_app()
