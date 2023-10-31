
import matplotlib.pyplot as plt 

artery_names = ["Left Interior Carotid", "Right Interior Carotid", "Basilar", "Left Middle Cerebral", "Right Middle Cerebral"]
def choice2num(choice):
    if choice=='lica':
        value = 0
    elif choice=='rica':
        value = 1
    elif choice=='b':
        value=2 
    elif choice=='lmca':
        value=3   
    elif choice=='rmca':
        value=4
    return value   

def choicestr2int(choice):
    if choice=='s':
        return 1
    elif choice=='lica':
        return 2
    elif choice=='rica':
        return 3
    elif choice=='b':
        return 4
    elif choice=='lmca':
        return 5
    elif choice=='rmca':
        return 6
    
def choicestr2int_tissue(choice):
    if choice=='w':
        return 1
    elif choice=='g':
        return 2 
    elif choice=='m' or choice=='w + g' or choice=='w+g' or choice=='wg' or choice == 'w g' or choice == 'g + w' or choice == 'gw' or choice == 'g w' or choice == 'g+w':
        return 3  
        
def choice2type_tissue(choice):
    if choice == 'g':
        type='Grey Matter'
    elif choice == 'w':
        type='White Matter'          
    return type  


def choice2type(choice):
    if choice == 1:
        type='Vein'
        subtype = 'Sinus Sagittalis'
    else:
        type='Artery' 
        if choice== 2:
            subtype='Left Interior Carotid'
        elif choice==3:
            subtype='Right Interior Carotid' 
        elif choice== 4:
            subtype='Basilar'
        elif choice== 5:
            subtype='Left Middle Cerebral'
        elif choice== 6:
            subtype='Right Middle Cerebral'    
    return type, subtype   

def choice2subtype(choice):
    if choice == 'b':
        subtype = 'Basilar'
    elif choice == 'lica':
        subtype = 'Left Interior Carotid'
    elif choice == 'rica':
        subtype = 'Right Interior Carotid'  
    elif choice == 'lmca':
        subtype = 'Left Middle Cerebral'   
    elif choice == 'rmca':
        subtype = 'Right Middle Cerebral'            
    return type, subtype

def on_esc(event):
    if event.key == 'escape':
        plt.close(event.canvas.figure)
