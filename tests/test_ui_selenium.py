import json
import os
import shutil
import time
from pathlib import Path

import pytest

selenium = pytest.importorskip("selenium")

from selenium import webdriver
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


pytestmark = pytest.mark.ui


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = os.environ.get(
    "SELENIUM_BASE_URL",
    "http://localhost:8501",
)

MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        "model/sample_model.joblib",
    )
).resolve()

DATASET_PATH = Path(
    os.environ.get(
        "DATASET_PATH",
        "dataset/test_dataset.csv",
    )
).resolve()

UI_STATUS_PATH = Path(
    "runtime/ui_status.json"
).resolve()

SCREENSHOT_PATH = Path(
    "runtime/ui_failure.png"
).resolve()


# ============================================================
# DRIVER
# ============================================================

def make_driver():
    """
    Create Chrome WebDriver.

    Priority:
    1. CHROMEDRIVER_PATH environment variable
    2. chromedriver found in PATH
    3. Selenium Manager
    """

    _project_driver = Path(__file__).resolve().parent.parent / "drivers" / "chromedriver.exe"
    _fixed_driver = Path(r"C:\WebDriver\chromedriver.exe")

    driver_path = (
        os.environ.get("CHROMEDRIVER_PATH")
        or shutil.which("chromedriver")
        or (str(_project_driver) if _project_driver.is_file() else None)
        or (str(_fixed_driver) if _fixed_driver.is_file() else None)
    )

    options = webdriver.ChromeOptions()

    # Headless Chrome
    options.add_argument("--headless=new")

    # Window size
    options.add_argument("--window-size=1440,1000")

    # Stability options
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Avoid first-run Chrome problems
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-notifications")

    # Reduce automation-related noise
    options.add_argument("--disable-blink-features=AutomationControlled")

    if driver_path:
        print()
        print("Using ChromeDriver:")
        print(driver_path)
        print()

        if not Path(driver_path).is_file():
            pytest.fail(
                "CHROMEDRIVER_PATH points to a file that does not exist:\n"
                f"{driver_path}"
            )

        try:
            return webdriver.Chrome(
                service=Service(driver_path),
                options=options,
            )
        except WebDriverException as exc:
            pytest.fail(
                "ChromeDriver was found but Chrome could not start.\n\n"
                f"Driver: {driver_path}\n"
                f"Error: {exc}"
            )

    # Offline requirement: do not invoke Selenium Manager because it may try
    # to download a browser driver. Require a driver already installed locally.
    pytest.skip(
        "Offline Selenium test requires a local ChromeDriver. "
        "Set CHROMEDRIVER_PATH or place chromedriver.exe in drivers/."
    )


# ============================================================
# STATUS
# ============================================================

def write_status(status: str, message: str):
    UI_STATUS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    UI_STATUS_PATH.write_text(
        json.dumps(
            {
                "status": status,
                "message": message,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# DEBUGGING
# ============================================================

def save_debug_screenshot(driver):
    """
    Save screenshot when the UI test fails.
    """

    try:
        SCREENSHOT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        driver.save_screenshot(
            str(SCREENSHOT_PATH)
        )

        print()
        print("Failure screenshot saved to:")
        print(SCREENSHOT_PATH)
        print()

    except Exception:
        pass


def print_page_information(driver):
    """
    Print useful information about the page when a test fails.
    """

    try:
        print()
        print("========== SELENIUM DEBUG ==========")
        print(f"Current URL: {driver.current_url}")
        print(f"Page title : {driver.title}")

        text = driver.find_element(
            By.TAG_NAME,
            "body",
        ).text

        print()
        print("Visible page text:")
        print("------------------------------------")
        print(text[:8000])
        print("------------------------------------")
        print()

    except Exception as exc:
        print(
            f"Could not collect page information: {exc}"
        )


# ============================================================
# WAIT HELPERS
# ============================================================

def wait_for_page(driver, timeout=30):
    """
    Wait until the page has loaded enough for Selenium.
    """

    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script(
            "return document.readyState"
        ) in ("interactive", "complete")
    )


def wait_for_text(
    driver,
    text: str,
    timeout: int = 30,
):
    """
    Wait until text appears anywhere on the page.
    """

    xpath = (
        "//*[contains("
        "translate(normalize-space(.), "
        "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
        "'abcdefghijklmnopqrstuvwxyz'), "
        f"'{text.lower()}'"
        ")]"
    )

    return WebDriverWait(
        driver,
        timeout,
    ).until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                xpath,
            )
        )
    )


def text_exists(driver, text: str):
    """
    Check whether text exists without waiting.
    """

    try:
        body_text = driver.find_element(
            By.TAG_NAME,
            "body",
        ).text.lower()

        return text.lower() in body_text

    except Exception:
        return False


# ============================================================
# SIDEBAR
# ============================================================

def click_sidebar_option(
    driver,
    text: str,
    timeout: int = 30,
):
    """
    Click a sidebar option.

    Supports Streamlit radio buttons,
    buttons, labels and normal text elements.
    """

    print(f"Clicking sidebar option: {text}")

    # First try an exact text match inside sidebar
    exact_xpaths = [
        f"//aside//*[normalize-space()='{text}']",
        f"//*[contains(@data-testid, 'stSidebar')]//*[normalize-space()='{text}']",
        f"//label[normalize-space()='{text}']",
        f"//button[normalize-space()='{text}']",
    ]

    last_error = None

    for xpath in exact_xpaths:

        try:
            element = WebDriverWait(
                driver,
                5,
            ).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        xpath,
                    )
                )
            )

            driver.execute_script(
                "arguments[0].scrollIntoView({block:'center'});",
                element,
            )

            try:
                WebDriverWait(
                    driver,
                    5,
                ).until(
                    EC.element_to_be_clickable(
                        (
                            By.XPATH,
                            xpath,
                        )
                    )
                ).click()

            except Exception:
                driver.execute_script(
                    "arguments[0].click();",
                    element,
                )

            time.sleep(0.3)

            print(
                f"Sidebar option '{text}' clicked."
            )

            return

        except Exception as exc:
            last_error = exc

    # More flexible fallback
    fallback_xpath = (
        f"//*[contains("
        f"translate(normalize-space(.),"
        f"'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
        f"'abcdefghijklmnopqrstuvwxyz'),"
        f"'{text.lower()}'"
        f")]"
    )

    try:
        element = WebDriverWait(
            driver,
            timeout,
        ).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    fallback_xpath,
                )
            )
        )

        driver.execute_script(
            "arguments[0].scrollIntoView({block:'center'});",
            element,
        )

        try:
            element.click()
        except Exception:
            driver.execute_script(
                "arguments[0].click();",
                element,
            )

        time.sleep(0.3)

        print(
            f"Sidebar option '{text}' clicked."
        )

        return

    except Exception as exc:
        last_error = exc

    raise TimeoutException(
        f"Could not click sidebar option '{text}'. "
        f"Last error: {last_error}"
    )


# ============================================================
# FILE UPLOAD
# ============================================================

def upload_file(
    driver,
    path: Path,
    timeout: int = 30,
):
    """
    Upload a file through Streamlit's file uploader.
    """

    if not path.exists():
        pytest.fail(
            f"File does not exist:\n{path}"
        )

    print()
    print(f"Uploading file:")
    print(path)
    print()

    # Streamlit file uploader
    selectors = [
        "input[type='file']",
        "input[data-testid='stFileUploaderFile']",
    ]

    file_input = None

    for selector in selectors:

        try:
            file_input = WebDriverWait(
                driver,
                5,
            ).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        selector,
                    )
                )
            )

            if file_input:
                break

        except Exception:
            continue

    if file_input is None:
        raise TimeoutException(
            "Could not find Streamlit file upload input."
        )

    file_input.send_keys(
        str(path)
    )

    time.sleep(0.8)

    print("File uploaded to browser.")


# ============================================================
# BUTTON CLICK
# ============================================================

def click_button(
    driver,
    text: str,
    timeout: int = 30,
):
    """
    Click a button containing the supplied text.
    """

    print(
        f"Looking for button: {text}"
    )

    xpath = (
        "//button[contains("
        "translate(normalize-space(.),"
        "'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
        "'abcdefghijklmnopqrstuvwxyz'),"
        f"'{text.lower()}'"
        ")]"
    )

    try:

        button = WebDriverWait(
            driver,
            timeout,
        ).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    xpath,
                )
            )
        )

        driver.execute_script(
            "arguments[0].scrollIntoView({block:'center'});",
            button,
        )

        try:
            WebDriverWait(
                driver,
                10,
            ).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        xpath,
                    )
                )
            ).click()

        except Exception:
            driver.execute_script(
                "arguments[0].click();",
                button,
            )

        print(
            f"Button clicked: {text}"
        )

    except Exception as exc:
        raise TimeoutException(
            f"Could not click button '{text}'. "
            f"Error: {exc}"
        )


# ============================================================
# MAIN TEST
# ============================================================

def test_end_to_end_ui():

    driver = None

    current_step = "Starting"

    try:

        # ----------------------------------------------------
        # STEP 1 - Check files
        # ----------------------------------------------------

        current_step = "Checking model file"

        print()
        print("=" * 60)
        print("MODEL GUARD SELENIUM UI TEST")
        print("=" * 60)
        print()

        print(
            f"Model path   : {MODEL_PATH}"
        )

        print(
            f"Dataset path : {DATASET_PATH}"
        )

        print(
            f"Base URL     : {BASE_URL}"
        )

        if not MODEL_PATH.exists():
            pytest.fail(
                f"Model file not found:\n{MODEL_PATH}"
            )

        if not DATASET_PATH.exists():
            pytest.fail(
                f"Dataset file not found:\n{DATASET_PATH}"
            )

        print()
        print("Model file   : OK")
        print("Dataset file : OK")
        print()

        # ----------------------------------------------------
        # STEP 2 - Start Chrome
        # ----------------------------------------------------

        current_step = "Starting Chrome"

        print(
            "Starting Chrome WebDriver..."
        )

        driver = make_driver()

        driver.set_page_load_timeout(
            60
        )

        print(
            "Chrome started successfully."
        )

        # ----------------------------------------------------
        # STEP 3 - Open application
        # ----------------------------------------------------

        current_step = "Opening ModelGuard application"

        print()
        print(
            f"Opening: {BASE_URL}"
        )

        # Marker query param so app.py can tell this page load is Selenium
        # itself driving the browser, not a real user session. Without this,
        # the app's auto-run-on-page-load logic would see this session reach
        # the Validation page and chain into launching ANOTHER Selenium
        # subprocess from inside this one -- runaway recursive spawning.
        driver.get(
            BASE_URL
            + ("&" if "?" in BASE_URL else "?")
            + "selenium_driver=1"
        )

        wait_for_page(
            driver,
            timeout=30,
        )

        time.sleep(0.5)

        print(
            f"Page opened: {driver.current_url}"
        )

        # ----------------------------------------------------
        # STEP 4 - Check ModelGuard
        # ----------------------------------------------------

        current_step = "Waiting for ModelGuard"

        print(
            "Waiting for ModelGuard..."
        )

        wait_for_text(
            driver,
            "ModelGuard",
            timeout=40,
        )

        print(
            "ModelGuard page detected."
        )

        # ----------------------------------------------------
        # STEP 5 - MODEL
        # ----------------------------------------------------

        current_step = "Opening Model section"

        click_sidebar_option(
            driver,
            "Model",
            timeout=30,
        )

        time.sleep(0.5)

        current_step = "Uploading model"

        upload_file(
            driver,
            MODEL_PATH,
            timeout=30,
        )

        # Wait for model success.
        # Some versions of the application use slightly
        # different messages, so check several possibilities.

        print(
            "Waiting for model upload confirmation..."
        )

        model_success = False

        for _ in range(30):

            if text_exists(
                driver,
                "Model loaded successfully",
            ):
                model_success = True
                break

            if text_exists(
                driver,
                "Model loaded",
            ):
                model_success = True
                break

            if text_exists(
                driver,
                "successfully",
            ):
                model_success = True
                break

            time.sleep(0.5)

        if not model_success:

            # Give the UI a little more time
            time.sleep(1.5)

            if not text_exists(
                driver,
                "Model loaded",
            ):

                raise TimeoutException(
                    "Model upload confirmation "
                    "was not detected."
                )

        print(
            "Model upload confirmed."
        )

        # ----------------------------------------------------
        # STEP 6 - DATASET
        # ----------------------------------------------------

        current_step = "Opening Dataset section"

        click_sidebar_option(
            driver,
            "Dataset",
            timeout=30,
        )

        time.sleep(0.5)

        current_step = "Uploading dataset"

        upload_file(
            driver,
            DATASET_PATH,
            timeout=30,
        )

        print(
            "Waiting for dataset upload confirmation..."
        )

        dataset_success = False

        for _ in range(30):

            if text_exists(
                driver,
                "Dataset uploaded",
            ):
                dataset_success = True
                break

            if text_exists(
                driver,
                "Dataset loaded",
            ):
                dataset_success = True
                break

            if text_exists(
                driver,
                "uploaded",
            ):
                dataset_success = True
                break

            time.sleep(0.5)

        if not dataset_success:

            time.sleep(1.5)

            if not text_exists(
                driver,
                "uploaded",
            ):

                raise TimeoutException(
                    "Dataset upload confirmation "
                    "was not detected."
                )

        print(
            "Dataset upload confirmed."
        )

        # ----------------------------------------------------
        # STEP 7 - VALIDATION
        # ----------------------------------------------------

        current_step = "Opening Validation section"

        click_sidebar_option(
            driver,
            "Validation",
            timeout=30,
        )

        time.sleep(0.5)

        print(
            "Validation section opened."
        )

        # ----------------------------------------------------
        # STEP 8 - Find validation button
        # ----------------------------------------------------

        current_step = "Finding validation button"

        validation_button_text = (
            "Run ModelGuard Validation"
        )

        print(
            "Waiting for validation button..."
        )

        click_button(
            driver,
            validation_button_text,
            timeout=40,
        )

        # ----------------------------------------------------
        # STEP 9 - Validation results
        # ----------------------------------------------------

        current_step = "Waiting for Validation Results"

        print()
        print(
            "Validation started."
        )

        print(
            "Waiting for Validation Results..."
        )

        try:

            wait_for_text(
                driver,
                "Validation Results",
                timeout=90,
            )

        except TimeoutException:

            # Some versions may display result content
            # without the exact heading.

            if not (
                text_exists(
                    driver,
                    "Model Loading",
                )
                or text_exists(
                    driver,
                    "Model Inference",
                )
                or text_exists(
                    driver,
                    "ML Security Testing",
                )
            ):
                raise

        print(
            "Validation results detected."
        )

        # ----------------------------------------------------
        # STEP 10 - CORE STAGES
        # ----------------------------------------------------

        current_step = "Checking validation stages"

        print()
        print(
            "Checking validation stages..."
        )

        stages = [
            "Model Loading",
            "Dataset Upload & Validation",
            "Data Imputation",
            "Preprocessing Testing",
            "Model Inference",
            "Result Validation",
            "Robustness Testing",
            "Out-of-Distribution Detection",
            "Distribution Shift Detection",
            "Data Poisoning Detection",
            "Model Poisoning Detection",
        ]

        for stage in stages:

            print(
                f"Checking: {stage}"
            )

            try:

                wait_for_text(
                    driver,
                    stage,
                    timeout=60,
                )

                print(
                    f"PASS - {stage}"
                )

            except TimeoutException:

                # Give Streamlit another few seconds
                time.sleep(5)

                if text_exists(
                    driver,
                    stage,
                ):
                    print(
                        f"PASS - {stage}"
                    )
                else:
                    raise TimeoutException(
                        f"Validation stage not found: "
                        f"{stage}"
                    )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        current_step = "Completed"

        write_status(
            "PASS",
            "Selenium end-to-end UI validation passed.",
        )

        print()
        print(
            "=" * 50
        )
        print(
            "SELENIUM UI TEST PASSED"
        )
        print(
            "=" * 50
        )
        print()

    except Exception as exc:

        # ----------------------------------------------------
        # FAILURE
        # ----------------------------------------------------

        error_message = (
            f"Selenium UI validation failed "
            f"during step: {current_step}. "
            f"Error: {exc}"
        )

        write_status(
            "FAIL",
            error_message,
        )

        print()
        print(
            "=" * 60
        )
        print(
            "SELENIUM UI TEST FAILED"
        )
        print(
            "=" * 60
        )
        print(
            f"Failed step: {current_step}"
        )
        print(
            f"Error: {exc}"
        )
        print(
            "=" * 60
        )
        print()

        if driver is not None:

            save_debug_screenshot(
                driver
            )

            print_page_information(
                driver
            )

        raise

    finally:

        if driver is not None:

            try:
                driver.quit()
            except Exception:
                pass