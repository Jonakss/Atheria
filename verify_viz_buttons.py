import time
from playwright.sync_api import sync_playwright

def verify(page):
    print("Navigating...")
    # Allow some time for server startup
    time.sleep(5)
    try:
        page.goto("http://localhost:3000/Atheria/", timeout=60000)
    except Exception as e:
        print(f"Navigation failed: {e}")
        return

    print("Waiting for dashboard...")
    # Wait for the "Modos de Vista" text or similar to ensure loading
    try:
        page.wait_for_selector("text=Modos de Vista", timeout=30000)
    except:
        print("Timeout waiting for 'Modos de Vista'. Taking screenshot for debug.")
        page.screenshot(path="verification_debug.png")
        raise

    print("Checking buttons...")
    # Check for Interferencia button
    interferencia = page.locator("text=Interferencia")
    if interferencia.count() > 0:
        print("Found Interferencia button!")
        interferencia.first.scroll_into_view_if_needed()
    else:
        print("Interferencia button NOT found.")

    # Check for Orbitales button
    orbitales = page.locator("text=Orbitales")
    if orbitales.count() > 0:
        print("Found Orbitales button!")
        orbitales.first.scroll_into_view_if_needed()
    else:
        print("Orbitales button NOT found.")

    time.sleep(1)

    print("Taking screenshot...")
    page.screenshot(path="verification_viz_buttons.png")

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # Set viewport large enough
        page.set_viewport_size({"width": 1280, "height": 800})
        try:
            verify(page)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()
