from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    # Start the frontend server first
    # I'll assume it's running on localhost:3001
    page.goto("http://localhost:3001/")

    # Wait for the login form to be visible
    page.wait_for_selector('form')

    page.screenshot(path="jules-scratch/verification/verification.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)
