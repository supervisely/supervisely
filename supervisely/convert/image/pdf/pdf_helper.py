from logging import Logger
from pathlib import Path
from typing import Tuple, Union


def pages_to_images(
    doc_path: Union[Path, str],
    save_path: Union[Path, str],
    dpi: int,
    logger: Logger,
    filetype: str = "pdf",
) -> bool:
    import fitz # pylint: disable=import-error

    doc_path = Path(doc_path)
    save_path = Path(save_path)

    name, is_pdf = get_file_name_and_check_type(doc_path, filetype)

    if not is_pdf:
        logger.warning(
            f"The file extention for document {name} is wrong or there is no extension founded."
        )
        logger.warning("Trying to read as PDF.")

    try:
        doc = fitz.Document(filename=doc_path, filetype=filetype)
    except:
        logger.warn(f"Skip converting {name} file: file is broken or have wrong extension.")
        return False

    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        pix_save_path = save_path / f"{name}_page_{page.number}.png"
        pix.save(pix_save_path)
    return True


def get_file_name_and_check_type(doc_path: Path, filetype: str = "pdf") -> Tuple[str, bool]:
    suffix = doc_path.suffix
    if len(suffix) > 0:
        suffix = suffix[1:]

    if suffix == filetype:
        return doc_path.stem, True
    else:
        return doc_path.name, False
