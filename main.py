from config.settings import *
from modules import *

def main():
    
    #Handle log_number choice and data directories
    log_number = select_log_number()
    data_directory, analysis_directory, nifti_directory, image_directory = setup_directories(log_number)
    parrec2nifti(data_directory, nifti_directory)
   
    #Welcome screen
    while True:
        welcome_screen()
        choice = welcome_screen_choice()

        #choices
        if choice == 0: #Show MRI images: DCE, Saggital T1/T2, Axial T1/T2
            viewer = MRIViewer(nifti_directory)
            viewer.display()
        #elif choice == 1:
            #do something 
        #elif choice == 2:
            #do something 
        #elif choice == 3:
            #do something 
        #elif choice == 4:
            #do something
        #elif choice == 5:
            #do something 
        #elif choice == 6:
            #do something  
        #elif choice == 7:
            #do something
        #elif choice == 8:
            #do something
        elif choice == 9:
            break       

if __name__ == '__main__':
    main()
