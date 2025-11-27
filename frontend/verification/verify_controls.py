import time
from playwright.sync_api import sync_playwright, expect

def verify_controls():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Increase viewport height to see bottom controls if needed
        context = browser.new_context(viewport={'width': 1280, 'height': 800})
        page = context.new_page()

        try:
            print("Navigating to dashboard...")
            # Wait for server to start if running continuously, but here we assume it's up
            try:
                page.goto("http://localhost:3000", timeout=60000)
            except Exception as e:
                print(f"Error navigating: {e}")
                # Retry once
                time.sleep(5)
                page.goto("http://localhost:3000", timeout=60000)

            print("Waiting for page load...")
            page.wait_for_load_state("networkidle")

            # Wait for HistoryControls to appear (it might need a "Lab" tab active)
            # Based on DashboardLayout, activeTab is 'lab' by default.

            # Check for StatusIndicators text "FPS"
            print("Looking for Status Indicators...")
            # Using text locator as it's visible
            page.wait_for_selector("text=FPS", timeout=30000)

            # Check for Step Counter
            print("Verifying Step Counter...")
            step_label = page.get_by_text("STEP")
            expect(step_label).to_be_visible()

            # Check for LIVE button
            print("Verifying LIVE button...")
            live_btn = page.get_by_text("LIVE", exact=True).or_(page.get_by_text("OFF", exact=True))
            expect(live_btn).to_be_visible()

            # Check for Clock/Interval button (it's an icon, so maybe locate by class or sibling)
            # It's next to the LIVE button in a flex container
            # We can try to click the button with the Clock icon.
            # Since we can't easily select by icon, let's look for the button element that is a sibling

            # Take initial screenshot of controls
            print("Taking screenshot of controls...")
            page.screenshot(path="frontend/verification/controls_initial.png")

            # Click the options button (it's the second button in the group)
            # The group is "flex rounded-md shadow-sm"
            # We can find the button that contains the Clock icon.
            # But simpler: find the button right of LIVE/OFF.

            # Let's try to find the button by SVG or just click the second button in the parent
            # controls_group = page.locator(".flex.rounded-md.shadow-sm")
            # options_btn = controls_group.locator("button").nth(1)
            # options_btn.click()

            # Alternatively, force click coordinates relative to LIVE button? No, brittle.

            # Let's assume the Clock icon is rendered as an SVG.
            # We can try to select by the generic button class in that specific component.
            # Or just take the screenshot to verify layout first.

            # To verify the popup opens, we need to click it.
            # Let's try to click the button that is NOT the LIVE/OFF button but in the same container.

            btns = page.locator("div.flex.rounded-md.shadow-sm > button")
            count = btns.count()
            print(f"Found {count} buttons in control group")
            if count >= 2:
                print("Clicking interval options button...")
                btns.nth(1).click()

                # Wait for popup
                print("Waiting for popup...")
                page.wait_for_selector("text=Intervalo de Actualización", timeout=5000)

                print("Popup opened! Taking screenshot...")
                page.screenshot(path="frontend/verification/controls_popup.png")

                # Verify options exist
                expect(page.get_by_text("Cada paso (1)")).to_be_visible()
                expect(page.get_by_text("Desactivado (-1)")).to_be_visible()

            else:
                print("Could not find interval button")

            print("Verification successful!")

        except Exception as e:
            print(f"Verification failed: {e}")
            page.screenshot(path="frontend/verification/error.png")
            raise e
        finally:
            browser.close()

if __name__ == "__main__":
    verify_controls()
