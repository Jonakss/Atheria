
import time
from playwright.sync_api import sync_playwright

def verify_controls():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = context.new_page()

        try:
            print("Navigating to dashboard...")
            # Navigate to the frontend (assuming port 3000 based on memory)
            page.goto("http://localhost:3000")

            # Wait for the app to load (MetricsBar should be visible)
            page.wait_for_selector("text=ATHERIA_LAB", timeout=10000)

            # Check for the MetricsBar (Controls section)
            # The play button should be visible in 'controls' mode
            print("Checking Controls Mode...")
            page.wait_for_selector("button:has-text('RUN')", timeout=5000)
            page.wait_for_selector("input[type='range']", timeout=5000) # Timeline should be visible

            # Take screenshot of Controls Mode
            page.screenshot(path="frontend/verification/controls_mode.png")
            print("Captured controls_mode.png")

            # Find the toggle button (ChevronRight - title 'Show Logs')
            print("Switching to Logs Mode...")
            toggle_btn = page.get_by_title("Show Logs")
            toggle_btn.click()

            # Wait for animation
            time.sleep(1)

            # Check for Logs Mode
            # Timeline slider should be GONE (or hidden)
            # Logs area should be expanded
            # Wait for animation by waiting for the timeline to disappear
            page.wait_for_selector("input[type='range']", state='hidden', timeout=5000)

            toggle_btn_back = page.get_by_title("Show Controls")
            if toggle_btn_back.is_visible():
                print("Toggle button changed state correctly.")

            # Take screenshot of Logs Mode
            page.screenshot(path="frontend/verification/logs_mode.png")
            print("Captured logs_mode.png")

        except Exception as e:
            print(f"Verification failed: {e}")
            page.screenshot(path="frontend/verification/error.png")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_controls()
