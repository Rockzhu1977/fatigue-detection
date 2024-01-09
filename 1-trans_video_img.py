import cv2
import os

os.chdir("G:\\Study\\fatiguedetector\\NTHUDDD\\Training_Evaluation_Dataset\\Training Dataset")

directory = r'G:\\Study\\fatiguedetector\\Dataset\\'
for files in os.listdir(os.path.join(directory, "video")):
    ############ Opening text files to label each frame #################
    txt_file = str(directory) + '\\label\\' + str(files[:-4]) + '_drowsiness.txt'
    file = open(txt_file, 'r') 
    n = 0
    temp = []
    while True: 
        # read by character 
        char = file.read(1)           
        if not char:  
            break
        n = n+1
        #print(char)
        temp.append(char)

    file.close()
    # Read the video from specified path 
    cam = cv2.VideoCapture(str(directory) + '\\video\\' + files) 

    # frame 
    currentframe = 0

    while(True): 
        # reading from frame 
        ret,frame = cam.read()
        if ret: 
            # if video is still left continue creating images 
            if temp[currentframe] == '1':
                # creating a folder named data 
                if files.find('yawn') != -1:
                    if not os.path.exists('train_data/images/3'): 
                        os.makedirs('train_data/images/3')
                    name = './train_data/images/3/' + str(files[:-4]) + '_' +str(currentframe) + '_' + 'drowsy' + '.jpg'
                else:
                    if not os.path.exists('train_data/images/2'): 
                        os.makedirs('train_data/images/2')
                    name = './train_data/images/2/' + str(files[:-4]) + '_' +str(currentframe) + '_' + 'drowsy' + '.jpg'
            
            if temp[currentframe] == '0':
                # creating a folder named data 
                if not os.path.exists('train_data/images/1'): 
                    os.makedirs('train_data/images/1')
                name = './train_data/images/1/' + str(files[:-4]) + '_' +str(currentframe) + '_' + 'notdrowsy' + '.jpg'
                
            print ('Creating...' + name) 
            cv2.imwrite(name, frame) 

            # increasing counter so that it will 
            # show how many frames are created 
            currentframe += 1
        else: 
            break

    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 
            
###################################################################################################
