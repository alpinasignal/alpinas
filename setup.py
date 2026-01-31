"""
Setup script for Alpina Signal
Run this after installing dependencies to set up the system
"""

import os
import sys
import subprocess


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    required = ["torch", "numpy", "pandas", "fastapi", "telegram"]

    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} not found")

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False

    return True


def create_directories():
    """Create necessary directories"""
    dirs = ["models", "data_cache", "logs"]

    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}/")


def check_env_file():
    """Check if .env file exists"""
    if os.path.exists(".env"):
        print("✓ .env file found")
        return True
    else:
        print("⚠ .env file not found")
        print("  Creating from .env.example...")

        if os.path.exists(".env.example"):
            with open(".env.example", "r") as f:
                content = f.read()

            with open(".env", "w") as f:
                f.write(content)

            print("✓ Created .env file")
            print("  Please edit .env and add your TELEGRAM_BOT_TOKEN")
            return False

        print("❌ .env.example not found")
        return False


def test_imports():
    """Test if critical imports work"""
    print("Testing imports...")

    try:
        import config
        print("✓ config.py")

        from data.market import BinanceDataFetcher
        print("✓ data.market")

        from data.features import FeatureEngine
        print("✓ data.features")

        from ai.model import create_model
        print("✓ ai.model")

        print("\n✓ All imports successful")
        return True

    except Exception as e:
        print(f"\n❌ Import error: {e}")
        return False


def main():
    """Main setup function"""
    print_header("Alpina Signal - Setup")

    print("Checking system requirements...\n")

    # Check Python version
    if not check_python_version():
        return

    print("\n" + "-"*60 + "\n")

    # Check dependencies
    if not check_dependencies():
        print("\n⚠ Please install dependencies first:")
        print("   pip install -r requirements.txt")
        return

    print("\n" + "-"*60 + "\n")

    # Create directories
    print("Creating directories...\n")
    create_directories()

    print("\n" + "-"*60 + "\n")

    # Check .env file
    print("Checking environment configuration...\n")
    env_ok = check_env_file()

    print("\n" + "-"*60 + "\n")

    # Test imports
    if not test_imports():
        return

    print("\n" + "="*60)
    print("  Setup Complete!")
    print("="*60 + "\n")

    if not env_ok:
        print("⚠ Next steps:")
        print("   1. Edit .env file and add your TELEGRAM_BOT_TOKEN")
        print("   2. Run: python main.py download")
        print("   3. Run: python main.py train-single --symbol BTCUSDT --timeframe 1h")
        print("\n   See QUICKSTART.md for detailed instructions")
    else:
        print("✓ System ready!")
        print("\n  Quick start commands:")
        print("   python main.py download          # Download data")
        print("   python main.py train-single --symbol BTCUSDT --timeframe 1h")
        print("   python main.py predict --symbol BTCUSDT --timeframe 1h")
        print("   python main.py api               # Start API server")
        print("   python main.py bot               # Start Telegram bot")
        print("\n  See README.md for full documentation")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
