import sys
from PySide6.QtWidgets import QApplication

# Import your custom window class with the correct capitalization
from digiled import DigitizerWindow

def main():
    # 1. Create the application instance (required for any PySide6 UI)
    app = QApplication(sys.argv)

    # 2. Create and set up the main window
    window = DigitizerWindow()
    window.resize(1850, 980)  # Setting a good default size for all those curves

    # 3. Tell the window to display itself
    window.show()

    # 4. Start the application's event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()