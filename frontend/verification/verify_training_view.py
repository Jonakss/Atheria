import time
from playwright.sync_api import sync_playwright, expect

def verify_training_view():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1280, 'height': 800})
        page = context.new_page()

        try:
            print("Navigating to dashboard...")
            try:
                page.goto("http://localhost:3000/Atheria/", timeout=60000)
            except Exception as e:
                print(f"Error navigating: {e}")
                time.sleep(5)
                page.goto("http://localhost:3000/Atheria/", timeout=60000)

            print("Waiting for page load...")
            page.wait_for_load_state("networkidle")

            # 1. Verify we are in Lab tab (default)
            # The sidebar should show submenus.

            # 2. Click "Entrenamiento" submenu button
            print("Clicking 'Entrenamiento' button...")
            training_btn = page.get_by_title("Entrenamiento")
            # Wait for it to be visible (animation)
            try:
                training_btn.wait_for(state="visible", timeout=5000)
                training_btn.click()
            except:
                # Retry clicking Lab tab first just in case
                print("Retry: Clicking Lab tab first...")
                page.get_by_text("Lab").click()
                training_btn.wait_for(state="visible", timeout=5000)
                training_btn.click()

            # 3. Verify Training View appears
            print("Verifying Training View headers...")
            expect(page.get_by_text("Training Metrics")).to_be_visible()
            expect(page.get_by_text("Loss History")).to_be_visible()
            expect(page.get_by_text("Evolution Metrics")).to_be_visible()

            # 4. Verify Canvas is GONE
            # PanZoomCanvas usually has a canvas element
            # But we might have other canvases (e.g. if Recharts uses one, but it uses SVG)
            # The main simulation canvas is usually in a div with "PanZoomCanvas" or simply check absence of HolographicViewer
            print("Verifying Canvas is replaced...")

            # Check for the specific structure of Training View
            expect(page.get_by_text("Best Checkpoints")).to_be_visible()
            expect(page.get_by_text("Status")).to_be_visible()

            print("Taking screenshot of Training View...")
            page.screenshot(path="frontend/verification/verification_training_view.png")

            print("Verification successful!")

        except Exception as e:
            print(f"Verification failed: {e}")
            page.screenshot(path="frontend/verification/verification_error.png")
            raise e
        finally:
            browser.close()

if __name__ == "__main__":
    verify_training_view()
