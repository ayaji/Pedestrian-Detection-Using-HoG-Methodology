#Folder structure of the project:
--------------------------------------------------------------------------------------------------------------------------
We have four folders:
1. Dataset - It is a folder which contains predefined dataset.
2. Descriptor - It is a folder where descriptors for positive and negative samples are calculated and saved.
3. Detector - It is a folder, where HOG Detectors are calculated and saved.
4. result_fig - This folder will have image detected results.

----------------------------------------------------------------------------------------------------------------------------
#Instruction to run the code:
---------------------------------------------------------------------------------------------------------------------------
Step 1: First run pdtools.py using command 'python pdtools.py'
Step 2: After 'pdtools.py' execution, run 'runDetection.py'
Step 2.1: We can run 'runDetection.py' in two ways:
			Step 2.1.1: With argument, which takes image input from the user and performs detection on it.
					a.) To run 'runDetection.py' with argument, use command 'python runDetection.py'
					b.) After executing the above cmd, it asks for the manual entry of the image path (ex: ..\data\pd1.jpg or C:\Users\adityayaji\Desktop\Assignments\CV\Project\data\pd1.jpg)
					    
			Step 2.1.2: Without argument, which takes images from the predefined image dataset mentioned in the folder 'Dataset' and performs detection on it.
					To run 'runDetection.py' without argument use command 'python runDetection.py'.

-------------------------------------------------------------------------------------------------------------------------------
Note:
1.If we run the code without argument, system will run detection on 40 images and produce results.
2.Enter 'h' for help instruction
3.Enter 'i' for inputting an image 
4.enter 'v' to see the detection on validation image set
5.Enter 'q' to exit
--------------------------------------------------------------------------------------------------------------------------------

