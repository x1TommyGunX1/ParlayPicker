import sys
from sports_betting import SportsBettingBot
from sports_betting_trainer import train_with_existing_data
from xrp_accumulator import XRPAccumulatorBot  # Assuming this exists
from config import *  # Import shared constants

def display_menu():
    """Display the main menu options."""
    print(f"╔════════════════════════════════════╗")
    print(f"║ Degen Package Menu v{PACKAGE_VERSION}         ║")
    print(f"║ 1. Train the Sports Betting Model  ║")
    print(f"║ 2. Run the Sports Betting Bot      ║")
    print(f"║ 3. Run the XRP Accumulator         ║")
    print(f"║ 4. Exit                            ║")
    print(f"╚════════════════════════════════════╝")

def main():
    """Main function to handle user input and execute selected options."""
    while True:
        display_menu()
        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            print("Training the sports betting model...")
            success = train_with_existing_data()
            if success:
                print("Training completed successfully!")
            else:
                print("Training failed. Check the log for details.")

        elif choice == '2':
            print("Starting the sports betting bot...")
            bot = SportsBettingBot()
            bot.run()

        elif choice == '3':
            print("Starting the XRP accumulator...")
            try:
                bot = XRPAccumulatorBot()
                bot.run()
            except Exception as e:
                print(f"Error running XRP accumulator: {e}")

        elif choice == '4':
            print("Exiting Degen Package...")
            sys.exit(0)

        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()