import sys
from general_analysis import general
from custom_analysis import custom

def general_analysis():
    general()

def custom_analysis():
    custom()

def exit_program():
    print("Close programm...")
    sys.exit()

def main_menu():
    while True:
        print("\nMenu:")
        print("1. General analysis.")
        print("2. Exit.")
        
        choice = input("Выберите вариант: ")

        if choice == '1':
            general_analysis()
        elif choice == '2':
            exit_program()
        else:
            print("Wrong input. Try again.")

if __name__ == "__main__":
    main_menu()
