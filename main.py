from config.settings import *
from modules import *

def main():
    #Handle log_number choice and welcome screen choice
    log_number = select_log_number()
    setup_directories(log_number)
    welcome_screen()
    choice = welcome_screen_choice()
    if choice == 0:
        #do something
    elif choice == 1:
        #do something 
    elif choice == 2:
        #do something 
    elif choice == 3:
        #do something 
    elif choice == 4:
        #do something
    elif choice == 5:
        #do something 
    elif choice == 6:
        #do something  
    elif choice == 7:
        #do something
    elif choice == 8:
        #do something
    elif choice == 9:
        #do something       

if __name__ == '__main__':
    main()
