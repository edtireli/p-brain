import os
import time

def add_notes(analysis_directory):
    user_text = input('[!] Enter notes: ')
    notes_path = os.path.join(analysis_directory, 'analysis_notes.txt')
    if not user_text.endswith('. '):
        if user_text.endswith('.'):
            user_text += ' '
        else:
            user_text += '. '
    with open(notes_path, 'a') as f:
        f.write(user_text)
    print('[!] Notes appended!')
    time.sleep(2)
