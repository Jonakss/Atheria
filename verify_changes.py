from playwright.sync_api import sync_playwright, expect
import time

def verify_lab_navigation():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Set viewport to ensure modal fits
        page = browser.new_page(viewport={'width': 1280, 'height': 800})

        try:
            # 1. Navigate to the app
            page.goto("http://localhost:4173")

            # Wait for main content to load
            page.wait_for_selector('text=ATHERIA')

            # 2. Click on 'Lab' tab
            lab_tab = page.locator('button[aria-label="Lab"]').first
            if not lab_tab.is_visible():
                # Try finding by icon or text if aria-label is missing, though NavButton usually has tooltip/title
                # Based on code: NavButton has title={label}
                lab_tab = page.locator('button[title="Lab"]')

            lab_tab.click()
            time.sleep(1) # Wait for animation
            lab_tab.click()
            # Wait for animation by waiting for the target element to become visible
            expect(page.locator('button[title="Inferencia"]')).to_be_visible()

            # 3. Verify Submenu appears (Inferencia, Entrenamiento, Análisis)
            inference_btn = page.locator('button[title="Inferencia"]')
            expect(inference_btn).to_be_visible()
            expect(inference_btn).to_be_visible()

            # Take screenshot of open menu
            page.screenshot(path="verification_lab_menu.png")
            print("Screenshot saved: verification_lab_menu.png")

            # 4. Click on 'Entrenamiento' sub-item
            training_btn = page.locator('button[title="Entrenamiento"]')
            training_btn.click()
            time.sleep(1)

            # 5. Verify LabSider shows Training section
            # The header in LabSider should say 'ENTRENAMIENTO'
            header = page.locator('span', has_text="ENTRENAMIENTO").first
            expect(header).to_be_visible()

            page.screenshot(path="verification_training_section.png")
            print("Screenshot saved: verification_training_section.png")

            # 6. Open Settings Modal
            # Find settings button in header (Settings icon)
            settings_btn = page.locator('button[title="Configuración Global"]')
            settings_btn.click()
            time.sleep(1)

            # 7. Verify Modal is visible and not cut off
            modal_title = page.locator('h2', has_text="CONFIGURACIÓN GLOBAL")
            expect(modal_title).to_be_visible()

            # Check if modal is centered.
            # We can just take a screenshot to verify visually.
            page.screenshot(path="verification_settings_modal.png")
            print("Screenshot saved: verification_settings_modal.png")

        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path="verification_error.png")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_lab_navigation()
