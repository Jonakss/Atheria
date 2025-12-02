
import time
from playwright.sync_api import sync_playwright

def verify_quantum_features():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Increase viewport size to capture more detail
        page = browser.new_page(viewport={'width': 1280, 'height': 800})

        try:
            # 1. Navigate to the application
            print("Navigating to application...")
            # Assuming backend is running on port 8000 and serving frontend assets
            # If backend is not serving frontend, we might need to run frontend dev server.
            # Based on memory, "To run the full development environment... backend is started with python run_server.py... frontend with npm run dev".
            # I should assume servers are running or start them.
            # But I cannot start long-running servers easily and wait for them in this script.
            # I will try to hit the backend URL first.

            # NOTE: I need to ensure the servers are running.
            # I will start them in background in a separate step if needed.
            # For now let's assume I need to start them.

            page.goto("http://localhost:3000/Atheria/", timeout=10000)

            # Wait for loading
            time.sleep(5)

            # 2. Verify "Quantum Genesis" in LabSider (New Experiment)
            print("Verifying Quantum Genesis toggle...")
            # Open LabSider if not open? It usually is open or accessible.
            # Check for "Nuevo Experimento" text or similar.

            # Take screenshot of LabSider
            page.screenshot(path="frontend/verification/labsider_quantum.png")

            # Check for "Quantum Genesis (IonQ)" text
            if page.get_by_text("Quantum Genesis (IonQ)").is_visible():
                print("✅ Quantum Genesis toggle found.")
            else:
                print("❌ Quantum Genesis toggle NOT found.")

            # 3. Verify Hybrid Controls in HistoryControls
            print("Verifying Hybrid Controls...")
            # Look for the Zap icon button in the bottom control bar
            # It might be in 'HistoryControls' which is at the bottom

            # Click the hybrid button to open the popover
            hybrid_btn = page.locator("button[title='Hybrid Simulation Controls']")
            if hybrid_btn.is_visible():
                hybrid_btn.click()
                time.sleep(1)
                page.screenshot(path="frontend/verification/hybrid_controls_popover.png")

                if page.get_by_text("Hybrid Simulation").is_visible():
                     print("✅ Hybrid Simulation popover opened.")
                else:
                     print("❌ Hybrid Simulation popover did NOT open.")
            else:
                print("❌ Hybrid Controls button NOT found.")

        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path="frontend/verification/error.png")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_quantum_features()
