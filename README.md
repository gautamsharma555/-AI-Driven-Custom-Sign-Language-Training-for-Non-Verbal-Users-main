# GeXT
A CNN based fully fledged Hand gesture to text converter for deaf &amp; dumb.

*************DATASET LINK**************
If you dont want to generate your own dataset, then you can use our dataset.
https://drive.google.com/drive/folders/1ejRWKWzXoNih9MT2E8m19hFtVjB9XXCk?usp=sharing

Steps to run this code on your pc :

********* STEPS TO TRAIN YOUR OWN MODEL**********

Directory Structure
1. create a directory named "CODEEX"
2. Put All the codes is "code.zip" folder to into Group_12.
3. Create subdirectory Dataset as Group_12/Dataset.
4. Create 28 empty folders , with each folder having the name of our gesture (these are for training purpose)
   (Eg of folder names are Group_12/Dataset/Atrain ,Group_12/Dataset/Btrain..and so on , likewise our code has
    28 gestures)
5. Create 28 more empty folders , with each folder having the name of our getsure (these are for model testing 
   purpose) (Eg of folder names are Group_12/Dataset/Atest,Group_12/Dataset/Btest..and so on , likewise our code
   has 28 such gestures for testing)
6. Create a subdirectory TrainedModel as Group_12/TrainedModel.

Generating the dataset
1. install all the required packages mentioned in "requirements.txt" file.
2. now run PalmTracker.py 56 time , each time changing the name of folder & number of images inside our code
   (mentioned inside the code at top)
3. Start PalmTracker.py and wait for 5 sec for it to read the background , then press 's' to start capturing
   the gesture and continue to show the gesture until the code stops.
4. The captured gesture image will be saved inside Dataset/{folder_name_given_in_code}
5. Repeat these above steps 56 time to get the whole dataset.

To Train the model
1. Run ModelTrainer.ipynb to train the model by changing the hyper parameters appropriately.
2. Name the model inside ModelTrainer.ipynb.
2. The trained model will be saved with the same name in the "TrainedModel" folder (which is to be created by you) 

To Test the project
1. Enter the name of model to be loaded inside MyApp2.py in line 333.
2. Run MyApp2.py and wait for some time while the saved model loads up
3. Press enter to start the gui process.
4. wait for 5 sec for the code to read your background for background elimination.
5. press 's' to start the gesture prediction.
6. Now show the gestures to generate characters/sentences.
7. press "q" to stop the app.


******STEPS TO RUN THE TRAINED MODEL*******
1. create a directory named Group_12.
2. put "MyApp2.py" inside this directory.
3. create a subdirectory TrainedModel as Group_12/TrainedModel.
4. Extract the contents of TrainedModel.zip into the TrainedModel directory.
5. You can customize the mapping of gestures to text if you want to, inside MyAp2.py in line 59 to line 119 otherwise
   default mapping will be taken which is mentioned in doc as well as ppt.
6. Run MyApp2.py to start the project.
7. Press enter to start the gui process.
8. wait for 5 sec for the code to read your background for background elimination.
9. press 's' to start the gesture prediction.
10. Now show the gestures to generate texts.
11. press "q" to stop the app.
