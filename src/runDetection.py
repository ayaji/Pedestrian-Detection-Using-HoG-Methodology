# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import pdtools
from sklearn import svm
from sklearn import cluster
import matplotlib.pyplot as plt
import time
import sys



'''
This function is for extracting HOG descriptor of an image
  Input:
          1. im: A grayscale image in height x width.
          2. bins: The number of bins in histogram.
          3. cells: The number of pixels in a cell.
          4. blocks: The number of cells in a block.
  Output:
          1. HOGBlock: The HOG descriptor of the input image 
'''
def ExtractHOG(im, bins, cells, blocks):
    # Pad the im in order to make the height and width the multiplication of
    # the size of cells.
    height, width = im.shape[0], im.shape[1]

    padHeight = 0
    padWidth = 0
    if height % cells[0] != 0:
        padHeight = cells[0] - height % cells[0]

    if width % cells[1] != 0:
        padWidth = cells[1] - width % cells[1]

    im = np.pad(im, ((0, padHeight), (0, padWidth)), 'edge')
    height, width = im.shape[0], im.shape[1]
	
	
    #########################################################################
    # TODO 1: 
    #  Compute the vertical and horizontal gradients for each pixel. Put them 
    #  in gradY and gradX respectively. In addition, compute the angles (using
    #  atan2) and magnitudes by gradX and gradY, and put them in angle and 
    #  magnitude.
    ########################################################################
    hx = np.array([[-1, 0, 1]])
    hy = hx.transpose()
    gradX = np.zeros((height, width))
    gradY = np.zeros((height, width))
    angle = np.zeros((int(height), int(width)))
    magnitude = np.zeros((height, width))
    ###########################  Begin TODO 1 #################################
    gradX = cv2.filter2D(im, cv2.CV_64F, hx) #horizontal gradient, here not -1, the same depth as im, 8u
    gradY = cv2.filter2D(im, cv2.CV_64F, hy) #vertical gradient
    magnitude = np.sqrt(gradX**2 + gradY**2)
    angle = np.arctan2(gradY, gradX)
    ###########################  End TODO 1 ###################################



    #############################################################################
    # TODO 2: 
    #  Construct HOG for each cells, and put them in HOGCell. numberOfVerticalCell
    #  and numberOfHorizontalCell are the numbers of cells in vertical and 
    #  horizontal directions.
    #  You should construct the histogram according to the bins. The bins range
    #  from -pi to pi in this project, and the interval is given by
    #  (2*pi)/bins.
    ##############################################################################
    numberOfVerticalCell = int(height/cells[0])
    numberOfHorizontalCell =int(width/cells[1])
    HOGCell = np.zeros((numberOfVerticalCell, numberOfHorizontalCell, bins))

    ###########################  Begin TODO 2 #################################
    ang_step = 2*np.pi / bins;
    for ix in range(numberOfVerticalCell):
        for iy in range(numberOfHorizontalCell):
            cell_mag = magnitude[cells[0]*ix:cells[0]*ix+8, cells[1]*iy:cells[1]*iy+8]
            cell_ang = angle[cells[0]*ix:cells[0]*ix+8, cells[1]*iy:cells[1]*iy+8]
            for ibin in range(bins):
                HOGCell[ix, iy, ibin] += np.sum(cell_mag[np.where((cell_ang > (-np.pi + ibin*ang_step)) & (cell_ang <= (-np.pi + (ibin+1)*ang_step)))])
            
    ###########################  End TODO 2 ###################################



    ############################################################################
    # TODO 3: 
    #  Concatenate HOGs of the cells within each blocks and normalize them. 
    #  Please remember to involve the small constant epsilon to avoid "division
    #  by zero". 
    #  The result should be stored in HOGBlock, where numberOfVerticalBlock and
    #  numberOfHorizontalBlock are the number of blocks in vertical and
    #  horizontal directions
    ###############################################################################
    numberOfVerticalBlock = numberOfVerticalCell - 1
    numberOfHorizontalBlock = numberOfHorizontalCell - 1
    HOGBlock = np.zeros((numberOfVerticalBlock, numberOfHorizontalBlock, \
                         blocks[0]*blocks[1]*bins))
    epsilon = 1e-10
    
    ###########################  Begin TODO 3 #################################
    for ix in range(numberOfVerticalBlock):
        for iy in range(numberOfHorizontalBlock):
            HOGBlock[ix, iy, :] = np.concatenate((HOGCell[ix, iy, :], HOGCell[ix, iy+1, :], HOGCell[ix+1, iy, :], HOGCell[ix+1, iy+1, :]))
            #normalization
            HOGBlock[ix, iy, :] /= np.sqrt(cv2.norm(HOGBlock[ix, iy, :], cv2.NORM_L2)**2 + epsilon)
#            HOGBlock[ix, iy, :] /= (cv2.norm(HOGBlock[ix, iy, :], cv2.NORM_L1) + epsilon) #1
#            HOGBlock[ix, iy, :] = np.sqrt(HOGBlock[ix, iy, :] / (cv2.norm(HOGBlock[ix, iy, :], cv2.NORM_L1) + epsilon)) #2
#            HOGBlock[ix, iy, :] /= np.sqrt(cv2.norm(HOGBlock[ix, iy, :], cv2.NORM_L1) + epsilon) #1


    
    ###########################  End TODO 3 ###################################
    return HOGBlock
    
'''
  This function is for training multiple components of detector
  Input:
          1. positiveDescriptor: The HOG descriptors of positive samples.
          2. negativeDescriptor: The HOG descriptors of negative samples.
  Output:
          1. detectors: Multiple components of detector
'''
def TrainMultipleComponent(positiveDescriptor, negativeDescriptor):
    ##########################################################################
    # TODO 1: 
    #  You should firstly set the number of components, e.g. 3. Then apply
    #  k-means to cluster the HOG descriptors of positive samples into 
    #  'numberOfComponent' clusters.
    ##########################################################################
    numberOfComponent = 3

    ###########################  Begin TODO 1 #################################
#    pos_sample_num, feature_len = positiveDescriptor.shape
#    cluster_center = np.random.rand(numberOfComponent, feature_len) #initial start point
#    label_pos = np.zeros((pos_sample_num))
#                     
#    num_iter = 100
#    distances = np.zeros((pos_sample_num, numberOfComponent))
#    for i in xrange(num_iter):
#        for j in xrange(numberOfComponent):
#            distances[:, j] = np.sum((positiveDescriptor - cluster_center[j, :])**2, axis=1)
#        label_pos = np.argmin(distances, axis=1)
#        for j in xrange(numberOfComponent):
#            cluster_center[j, :] = np.mean(positiveDescriptor[label_pos == j, :], axis=0)

    cluster_centers, label_pos, best_inertia = cluster.k_means(positiveDescriptor, numberOfComponent)
    ###########################  End TODO 4 ###################################

    
    
    ############################################################################
    # TODO 2: 
    #  After TODO 1, you have 'numberOfComponent' positive sample clusters.
    #  Then you should use train 'numberOfComponent' detectors in this TODO.
    #  For example, if 'numberOfComponent' is 3, then the:
    #    1st detector should be trained with 1st cluster of positive samples vs
    #    all negative samples;
    #    2nd detector should be trained with 2nd cluster of positive samples vs
    #    all negative samples;
    #    3rd detector should be trained with 3rd cluster of positive samples vs
    #    all negative samples;
    #    ...
    #  To train all detectors, please use SVM toolkit such as sklearn.svm.
    detectors = [None] * numberOfComponent

    ###########################  Begin TODO 2 ###########################
    detectors = np.array(detectors)
    for i in range(numberOfComponent):
        svm_model = svm.LinearSVC()
        trainSample = np.concatenate((positiveDescriptor[i == label_pos], negativeDescriptor))
        label_neg_ = np.zeros((negativeDescriptor.shape[0]))
        label_pos_ = np.ones((label_pos[i == label_pos].shape[0]))
        label = np.concatenate((label_pos_, label_neg_), axis=0)
        svm_model.fit(trainSample, label)
        detectors[i] = svm_model.coef_
    
    
    ###########################  End TODO 2 ###########################
    return detectors    

'''
  This function is for multiscale detection
  Input:
          1. im: A grayscale image in height x width.
          2. detectors: The trained linear detectors. Each one is in 
                        row x col x dim. The third dimension 'dim' should 
                        be the same with the one of the HOG descriptor 
                        extracted from input image im.
          3. threshold: The constant threshold to control the prediction.
          4. bins: The number of bins in histogram.
          5. cells: The number of pixels in a cell.
          6. blocks: The number of cells in a block.
  Output:
          1. bbox: The predicted bounding boxes in this format (n x 5 matrix):
                                   x11 y11 x12 y12 s1
                                   ... ... ... ... ...
                                   xi1 yi1 xi2 yi2 si
                                   ... ... ... ... ...
                                   xn1 yn1 xn2 yn2 sn
                   where n is the number of bounding boxes. For the ith 
                   bounding box, (xi1,yi1) and (xi2, yi2) correspond to its
                   top-left and bottom-right coordinates, and si is the score
                   of convolution. Please note that these coordinates are
                   in the input image im.
'''
def MultiscaleDetection(im, detectors, threshold, bins, cells, blocks):
    #############################################################################
    # TODO 1: 
    #  You should firstly generate a series of scales, e.g. 0.5, 1, 2. And then
    #  resize the input image by scales and store them in the structure pyra.
    ############################################################################
    pyra = []
    scales = []

    ###########################  Begin TODO 1 ###########################
    scale_factor = [1.0]
    scales = np.divide(1, scale_factor)
    scaled_im = im
    for scale in scales:
        scaled_im = cv2.resize(im, (0,0), fx=scale, fy=scale)
        pyra.append(scaled_im)
        
    ###########################  End TODO 1 ###########################


    #############################################################################
    #  TODO 2:
    #  Perform detection on multiscale. Please remember to transfer the
    #  coordinates of bounding box according to their scales
    #############################################################################
    bbox = []
    numberOfScale = len(pyra)

    ###########################  Begin TODO 2 ###########################
    det_dim = detectors.shape[0]
    x1 = x2 = y1 = y2 = 0
    for i in range(numberOfScale):
        #step one, construct HOG descriptor using ExtractHOG(im, bins, cells, blocks)
        HOG_descriptor = ExtractHOG(pyra[i], bins, cells, blocks)
        #step two, convolve detectors on HOG descriptor
        for j in range(det_dim):
            for ix in range(HOG_descriptor.shape[0] +1 - detectors[j].shape[0]):
                for iy in range(HOG_descriptor.shape[1] +1 - detectors[j].shape[1]):
                    score = np.sum((HOG_descriptor[ix:ix+detectors[j].shape[0], iy:iy+detectors[j].shape[1],:]*detectors[j]))
                    if score >= threshold[i]:
                        x1 = ix * cells[0] 
                        y1 = iy * cells[1] 
                        x2 = x1 + 128
                        y2 = y1 + 64
                        box = np.array([y1/scales[i], x1/scales[i], y2/scales[i], x2/scales[i], score])
                        bbox.append(box)
#                        print score
            
    bbox = np.array(bbox)
	
    ###########################  End TODO 2 ###########################
    return bbox    
    
    
# Set the number of bin to 9
bins = 9

# Set the size of cell to cover 8 x 8 pixels
cells = [8, 8]

# Set the size of block to contain 2 x 2  cells
blocks = [2, 2]    
    
###################################################################
# Step 1: Extract HOG descriptors of postive and negative samples
###################################################################
pos = np.load('./Dataset/Train/pos.npy')
neg = np.load('./Dataset/Train/neg.npy')
numberOfPositive = pos.shape[3]
numberOfNegative = neg.shape[3]
height, width = pos.shape[0], pos.shape[1]

# Delete the descriptor files if you want to extract new ones
if os.path.exists('./Descriptor/positiveDescriptor.npy') and \
    os.path.exists('./Descriptor/negativeDescriptor.npy'):
        
    positiveDescriptor = np.load('./Descriptor/positiveDescriptor.npy')
    negativeDescriptor = np.load('./Descriptor/negativeDescriptor.npy')
else:
    positiveDescriptor = [None] * numberOfPositive
    for ii in range(0, numberOfPositive):
        print('Positive HOG descriptor: ' + str(ii + 1) + '\\' + str(numberOfPositive))
        temp = ExtractHOG(cv2.cvtColor(pos[:,:,:,ii], cv2.COLOR_BGR2GRAY), bins, cells, blocks)
        positiveDescriptor[ii] = temp.ravel().transpose()
    positiveDescriptor = np.array(positiveDescriptor)
    
    negativeDescriptor = [None] * numberOfNegative
    for ii in range(0, numberOfNegative):
        print('Negative HOG descriptor: ' + str(ii + 1) + '\\' + str(numberOfNegative))
        temp = ExtractHOG(cv2.cvtColor(neg[:,:,:,ii], cv2.COLOR_BGR2GRAY), bins, cells, blocks)
        negativeDescriptor[ii] = temp.ravel().transpose()   
    negativeDescriptor = np.array(negativeDescriptor)
    
    np.save('./Descriptor/positiveDescriptor.npy', positiveDescriptor)
    np.save('./Descriptor/negativeDescriptor.npy', negativeDescriptor)

########################################################################  
# Step 2: Train Linear Detector
########################################################################
# Delete the detector file if you want to train a new one
if os.path.exists('./Detector/detectors.npy'):
    detectors = np.load('./Detector/detectors.npy')
else:
    print('Training linear detector');
    detectors = TrainMultipleComponent(positiveDescriptor, negativeDescriptor);
    print(len(detectors))
    print(height)
    print(width)

    for ii in range(0, len(detectors)):
        detectors[ii] = np.reshape(detectors[ii], (height/cells[0] - 1, \
                                width/cells[1] - 1, blocks[0] * blocks[1] * bins))

    np.save('./Detector/detectors.npy', detectors)
    
#########################################################################    
# Step 3: Detection
#########################################################################
def user_input():
    print("Enter the path of the image")
    l=input()
    
    img=cv2.imread("%s"%(l),1)

    threshold1 = [1]
    print('Detecting')
    bbox = MultiscaleDetection(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), \
                               detectors, threshold1, bins, cells, blocks)

    top1 = np.arange(bbox.shape[0])   
    top1 = pdtools.NonMaxSup(bbox, 0.3)
    pdtools.ShowBoxes1(img, bbox[top1,:])
    
	
def validation_image():
    validation = np.load('./Dataset/Validation/validation.npy',encoding='latin1')
    groundTruth = np.load('./Dataset/Validation/groundTruth.npy',encoding='latin1')


    numPositives = 0;
    for ii in range(0, len(groundTruth)):
        numPositives = numPositives + groundTruth[ii].shape[0];

    Label = []
    Score = []
    threshold = [1]
    neg_cnt = 0
    t1 = time.time()

    for ii in range (0, len(validation)):
        print('Detect ' + str(ii) + '...')
        bbox = MultiscaleDetection(cv2.cvtColor(validation[ii], cv2.COLOR_BGR2GRAY), \
                                   detectors, threshold, bins, cells, blocks)  
        if bbox.shape[0] == 0:
            continue
        top = np.arange(bbox.shape[0])
        # Non-maximum suppression. Uncomment this line if you want this
        # process.
        top = pdtools.NonMaxSup(bbox, 0.3)
        print (bbox[top])

        # Measure the performance
        labels, scores = pdtools.MeasureDetection(bbox[top, :], groundTruth[ii], 0.5)
        Label += labels.tolist()
        Score += scores.tolist()

        # Show the bounding boxes. Uncomment the following two line if you 
        # want to show the bounding boxes.
        pdtools.ShowBoxes(validation[ii], bbox[top,:], ii)
        #raw_input("Press Enter to continue...")

    t2 = time.time()
    print (t2 - t1)
    Label = np.array(Label)
    Score = np.array(Score)
    Label = np.reshape(Label, (Label.shape[0]))
    Score = np.reshape(Score, (Label.shape[0]))
    ap = pdtools.DrawPRC(Label, Score, numPositives)

def key_selection():
    print('\n')
    print("1.Enter 'h' for help instruction")
    print("2.Enter 'i' for inputting an image ")
    print("3.enter 'v' to see the detection on validation image set")
    print("4.Enter 'q' to exit")

    m=input()
    if(m=='i'):
        user_input()
        key_selection()
          
          
    elif(m=='h'):
        print("'h' is pressed.")
        print("Pedestrian Detection")
        print("--------------------------Description----------------------------")
        print("We can Run runDetection.py in two different ways")
        print("1. With argument, which takes image input from the user and performs pedestrain detection on it.")
        print("----------------------->command to run with argument is 'python runDetection.py image location'")
        print("2. Without argument, which takes images from the predefined data set mentioned in the folder 'Dataset'")
        print("----------------------->command to run without argument is 'python runDetection.py'")
        print("Input")
        print("Set of Images as numpy array")
        print("Processing")
        print("1. Compute angle and magnitude of gradient for each pixel")
        print("2. Construct HOG for each cell")
        print("3. Construct the normalized HOG for each block")
        print("4. Convolute the learned detector on the HOG descriptor of the image")
        print("5. Select the convolution above the threshold and compute its bounding box")
        print("6. Build image pyramid")
        print("7. Perform detection on multiple scales of image")
        print("Output")
        print("Graph with bounding boxes detecting pedestrain in the image")
        print("-----------------Instructions to execute the program---------------------")
        print("First please run pdtools.py file using the command python pdtools.py")
        print("After running pdtools.py please run runDetection.py")

        key_selection()
          
          
    elif(m=='v'):
        validation_image()
        key_selection()
    
    elif(m=='q'):
        sys.exit()
          
key_selection()

