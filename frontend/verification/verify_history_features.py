
import time
from playwright.sync_api import sync_playwright

def verify_history_features():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Using a larger viewport to ensure sidebar and content are visible
        context = browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = context.new_page()

        try:
            print("Navigating to app at http://localhost:8000/index.html ...")
            # Point to backend server which serves the built frontend
            page.goto("http://localhost:8000/index.html", timeout=60000)

            print("Waiting for main content...")
            # Wait for the sidebar or any main element
            try:
                page.wait_for_selector("nav", timeout=30000)
                print("Navigation loaded.")
            except:
                print("Timeout waiting for nav. Dumping page content snippet:")
                print(page.content()[:500])
                # Proceed anyway to check if elements are present
                pass

            # Give it a moment to fully render and connect WS
            time.sleep(5)

            print("Checking for History UI elements...")

            # 1. Search Input
            search_input = page.locator("input[placeholder='Buscar experimento...']")
            if search_input.count() > 0:
                 print("✅ Search input found")
            else:
                 print("❌ Search input NOT found")

            # 2. Import Button (Text "Importar")
            import_btn = page.get_by_text("Importar")
            if import_btn.count() > 0:
                print("✅ Import button found")
            else:
                print("❌ Import button NOT found")

            # 3. Refresh Button (Title "Actualizar lista")
            refresh_btn = page.locator("button[title='Actualizar lista']")
            if refresh_btn.count() > 0:
                print("✅ Refresh button found")
            else:
                print("❌ Refresh button NOT found")

            # 4. Check for Delete Buttons (to imply Modal testability)
            delete_btns = page.locator("button[title='Eliminar']")
            if delete_btns.count() > 0:
                print(f"✅ Delete buttons found ({delete_btns.count()})")
            else:
                # If no experiments, we won't see delete buttons, which is expected
                print("ℹ️ No delete buttons found (likely no experiments list)")

            # Take screenshot for record
            page.screenshot(path="frontend/verification/history_features.png")
            print("Screenshot saved to frontend/verification/history_features.png")

        except Exception as e:
            print(f"Error during verification: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_history_features()
