# -*- coding: utf-8 -*-
"""
Define a PSFExtractor class that used to process astronomical images about .fit*.
Created on Saturday August 4 2019

@author: syy
"""

import sys,os,glob ,math,scipy,itertools
import numpy as np
from astropy.io import fits

class PSFExtractor():
    '''
    define a class to deal fits
    three mainly function:
        get the coordinates of stars in the fits
        get the SNR of the fits
        get the Ellipisicity of stars in the fits
    '''
    def __init__(self, path, save, mesh, mode = 8):
        self.path = path + '/*.fit*'
        self.save = save
        self.mesh = mesh
        # The connected domain's mode is 8.
        self.mode = mode
        # The number of fits in input path. The default is 0
        self.counts = 0
        # Value can get other number, let it equal to 3 in threshold of the project
        self.value = 3
        # Get the list of each fits' path
        self.fitfile = self.get_fitfile()
        # Determines the TXT file save path
        self.creat_save()

    # Main function used to get the list of each fits' path
    def get_fitfile(self):
        # Get the matching file path names one by one
        fitfile = glob.glob(self.path)
        # While there is no fits under the specified path
        while len(fitfile)==0:
            print("Error:File doesn't exist, please check the path.\n")
            path = input("Input correct path:")
            if path == " ":
                # Take the project path to find fits
                path = sys.path[0]
            else:
                return
            path += '/*.fit*'
            fitfile = glob.glob(path)
        return fitfile

    # Generates the TXT file for the location of the stars in the fits
    def get_location(self):
        for fit in self.fitfile:
            # return a 0-1 array of fits
            gray = self.deal_img(fit)
            # Get the right labels of every connected domain
            labels = self.get_label(gray)
            # Create a txt file to storage the positions of pixels what we extract
            self.creat_location(fit, labels, gray)

    # Get teh value, height and width of fits
    def get_fit(self, fit):
        self.counts += 1
        print(fit, self.counts)
        # open the fits file
        hdul = fits.open(fit)
        # Put the gray level imfornation in to matrix
        gray = hdul[0].data
        # Get the row number and column number
        h = hdul[0].header['NAXIS1']
        w = hdul[0].header['NAXIS2']
        # Transfer gray
        gray = gray.T
        return gray, h, w

    # Main function binarization of fits values
    def deal_img(self, fit):
        # Parameters's initialization
        mesh_h = 0
        mesh_w = 0
        i = 0
        j = 0
        # Get teh value, height and width of fits
        gray,h,w = self.get_fit(fit)
        # Make a copy in gray
        copy = gray.copy()
        # Get the value of batch size
        mesh = int(self.mesh)
        mesh_h = mesh + j
        mesh_w = mesh + i
        # Create a local area
        while mesh_h <= h:
            while mesh_w <= w:
                local_area = gray[i:mesh_w, j:mesh_h]
                # Compute the local area's all the pixels's mean,median,standard deviation
                mean = np.mean(local_area)
                median = np.median(local_area)
                std_d = np.std(local_area)
                # Set a threshold to remove some pixels
                threshold = median + self.value*std_d
                # Scan the local area's pixels to find useful pixels
                for x in range(i, mesh_w):
                    for y in range(j, mesh_h):
                        x_axis = x % mesh
                        y_axis = y % mesh
                        # Create a 0-1 array
                        if local_area[x_axis, y_axis] > threshold:
                            gray[x,y] = 1
                        else:
                            gray[x,y] = 0
                i += mesh
                mesh_w += mesh
            i = 0
            mesh_w = i + mesh
            j += mesh
            mesh_h += mesh
        # return a 0-1 array
        return  gray

    # Get the right labels of every connected domain
    def get_label(self,gray):
        # The data's type we input is array. To change it into a nested list.
        BW = list(gray)
        for k in range(len(BW)):
            BW[k] = list(BW[k])
        # Compute the number of Non-zero pixels which next to each other in each column
        numRuns = self.get_num_runs(BW)
        # Compute every run's start row,end row and current column
        sr, er, c = self.fill_run_vectors(BW)
        # Deal with all of Runs in a image.Define a label to every exit Run.
        labels, req, ceq = self.first_pass(numRuns, sr,er,c)
        # Get the final tag list
        labels = self.get_labels(labels, req, ceq)
        M, N = self.size(BW)
        L = scipy.zeros((M, N))
        for s, e, c, l in zip(sr, er, c, labels):
            L[c - 1, (s - 1):e] = l
        return L

    # Compute the IN's list's length and the list's first element's length
    def size(self,IN):
        # The length of the first element.if IN is a nested list, M represent the number of columns
        M = len(IN[0])
        # The length of the list.if IN is a nested list,N represent the number of row
        N = len(IN)
        return (M, N)

    # Compute the number of Non-zero pixels which next to each other in each column
    # Run means the Non-zero pixels
    def get_num_runs(self,IN):
        # M represent the number of columns,N represent the number of rows
        M, N = self.size(IN)
        #Set the init value of Runs
        result = 0
        if M != 0 and N != 0:
            # Perform column scanning on the image
            for col in IN:
                # First element is non-zero in a column?
                if col[0] != 0:
                    # If ture,the number of Runs + 1
                    result += 1
                # Scan all the columns
                for idx in range(1, M):
                    # The current element is non-zero and the last element is zero?
                    if col[idx] != 0 and col[idx - 1] == 0:
                        result += 1
        return result

    # Compute every run's start row,end row and current column
    def fill_run_vectors(self, IN):
        M, N = self.size(IN)
        sr = []  # sr[k] represent the first K Run's start row
        er = []  # er[k] represent the first k Run's end row
        c = []   #c[k]  represent the first k current column
        # Scan the whole image, cidx represent the iteration's time,col represent the element
        for cidx, col in enumerate(IN):
            # Stand at the front of each column
            k = 0
            # Search for the first 1 appear in a column
            while k < M:
                # try-except deal with the abnormity
                try:
                    # Move to the position of 1
                    k += col[k:].index(1)
                    c.append(cidx + 1)
                    sr.append(k + 1)
                    # Next to search for the first 0 appear in a column
                    try:
                        # Move to the position of 0
                        k += col[k:].index(0)
                    except ValueError:
                        # Break the recurrent
                        k = M
                    er.append(k)
                except ValueError:
                    break
        # Return every run's start row,end row and current column.
        return sr, er, c

    # Deal with all of Runs in a image.Define a label to every exit Run
    def first_pass(self, numRuns, sr, er, c):
        # Initialize
        currentColumn = 0
        nextLabel = 1
        firstRunOnPreviousColumn = -1
        lastRunOnPreviousColumn = -1
        firstRunOnThisColumn = -1
        # Create a empty list for storaging
        equivList = []
        # First create the number of Runs labels
        labels = [0] * numRuns
        # Judge the mode of connected domain
        if self.mode == 8:
            offset = 1
        else:
            offset = 0
        # Traveling all of Runs,we deal with the K Runs in every circulation
        for k in range(numRuns):
            # If The K Run and The K-1 Run are in adjacent columns
            if c[k] == currentColumn + 1:
                firstRunOnPreviousColumn = firstRunOnThisColumn
                # Let firstRunOnThisColumn point to The k Run ,The k Run became the first Run in the column
                firstRunOnThisColumn = k
                # Let lastRunOnPreviousColumn point to the k-1 Run,The k-1 Run became the last Run in the last column
                lastRunOnPreviousColumn = k - 1
                # Let currentColumn point to the k Run's column
                currentColumn = c[k]
            # If The K Run and The k-1 Run are not in adjacent columns
            elif c[k] > currentColumn + 1:
                # Represent not in adjacent column
                firstRunOnPreviousColumn = -1
                lastRunOnPreviousColumn = -1
                firstRunOnThisColumn = k
                currentColumn = c[k]
            else:
                pass
            # In the situation of  adjacent columns
            if firstRunOnPreviousColumn >= 0:
                p = firstRunOnPreviousColumn
                # Deal with all the Runs which next to the current the K Run
                while p <= lastRunOnPreviousColumn and sr[p] <= er[k] + offset:
                    # Judge if these Runs have the p Run and the current Run overlapped on the row
                    if er[k] >= sr[p] - offset and sr[k] <= er[p] + offset:
                        # If it is true,please define the same labels,if it is false,storage in the list of equivList
                        if labels[k] == 0:
                            labels[k] = labels[p]
                        else:
                            if labels[k] != labels[p]:
                                equivList.insert(0, (labels[k], labels[p]))
                            else:
                                pass
                    p += 1
            # Label the labels,start from 1
            if labels[k] == 0:
                labels[k] = nextLabel
                nextLabel += 1
        # Create two empty list
        rowEquivalences = []
        colEquivalences = []
        if len(equivList) > 0:
            # The list of equivList's element appear as one pair
            for item0, item1 in equivList:
                rowEquivalences.append(item0)
                colEquivalences.append(item1)
        return labels, rowEquivalences, colEquivalences

    # Solve the problem that The Runs belong to the same culster but have different labels
    def get_labels(self, labels, rowEq, colEq):
        lblist = []
        # Take a node in a for circular
        for k, r in enumerate(rowEq):
            if r == -1:
                continue
            queue = list([rowEq[k], colEq[k]])
            cur_lblist = list()
            # Start with a node which can reach at any node to find all the node
            while queue:
                head = queue.pop(0)
                # Package in the cur_lblist
                cur_lblist.append(head)
                for n in range(k + 1, len(rowEq)):
                    if rowEq[n] == head:
                        queue.append(colEq[n])
                        # The visited node set -1
                        rowEq[n] = -1
                        colEq[n] = -1
                    elif colEq[n] == head:
                        queue.append(rowEq[n])
                        rowEq[n] = -1
                        colEq[n] = -1
            # Storage in the lblist
            lblist.append(cur_lblist)
        labels = scipy.array(labels)
        # Transfer into a array
        for oldlabels in lblist:
            for ol in oldlabels:
                labels[labels == ol] = oldlabels[0]
        # zip()'s output type is tuple,so changed it into list to sort
        sort_labels = list(zip(labels, range(len(labels))))
        # After find out these Runs which belong to the same culster need to be sorted
        sort_labels.sort()
        sort_idx = [k[1] for k in sort_labels]
        sort_labels = scipy.array([k[0] for k in sort_labels])
        if sort_labels[0] != 1:
            sort_labels -= (sort_labels[0] - 1)
        for k in range(1, len(sort_labels)):
            cur_label = sort_labels[k]
            pre_label = sort_labels[k - 1]
            if cur_label > pre_label + 1:
                sort_labels[sort_labels == cur_label] = pre_label + 1
        for k, l in zip(sort_idx, sort_labels):
            labels[k] = l
        return labels

    # Create a txt file to storage the positions of pixels what we extract
    def creat_location(self,fit,matrix,orimat):
        orimat = orimat.copy()
        # Change the name of the saved file
        filepath = fit.split("\\")[-1]
        # Write a file
        file = open(self.save + '/'+ filepath +'.txt', 'w+')
        # Compute the max number of lable,the type is np.int64
        num = np.amax(matrix)
        count = 0
        tag = 1  # init the tag number
        while tag <= (num):
            obj = matrix.copy()
            # Scan the matrix's row and column
            for i in range(len(matrix[0])):
                for j in range(len(matrix)):
                    # Judge if find the connected domain,taat is to say the axis we index == tag
                    if obj[i, j] == tag:
                        # Assign the original pixel position
                        obj[i, j] = orimat[i, j]
                        count += 1
                    else:
                        obj[i, j] = 0
            # Delete regions too small to extract PSFs
            # Remove the noise
            if count >= 2:
                element = np.where(obj == np.max(obj))
                # Make sure the position of matrix's max value,xval represent the row value,yval represent the column value.
                xval = element[0][0] + 1
                yval = element[1][0] + 1
                # Write the noise's position into file.
                file.write("%s,%s\n" % (xval, yval))
            tag += 1

    # -----------------------------------------------------------------------------------------------
    # Determines the TXT file save path
    def creat_save(self):
        # If the folder exists, pass
        if os.path.isdir(self.save) == True:
            pass
        # While we input a empty folder, automatically save in the project path
        elif self.save == '':
            self.save = sys.path[0]
        else:
            #create corresponding file
            os.makedirs(self.save, 0o777)

    # -----------------------------------------------------------------------------------------------
    # Function definition SNR is used to calculate SNR of images
    def get_snr(self, sigmacut=1):
        # Ref to lucky imaging paper by Staley
        # https://www.ast.cam.ac.uk/sites/default/files/SPIE_7735-211_010710.pdf
        # Equation 5
        # Data reduction strategies for lucky imaging
        # img is the observational data 2D ndarray
        # sigmacut is the value used to cut background
        for fit in self.fitfile:
            hdul = fits.open(fit)
            img = hdul[0].data
            totalflux = np.sum(img)
            # Calculate the mean flux of all
            meanflux = np.mean(img)
            # Calculate the background with one sigma clipped
            imgsigma = np.var(img) ** 0.5 * sigmacut
            # Calculate the background contribution
            backtotal = np.mean(img[img <= imgsigma + meanflux]) * np.shape(img)[0] * np.shape(img)[1]
            # Calculate the background sigma contribution (background possion noise contribution)
            backsigma = np.var(img[img <= imgsigma + meanflux]) * np.shape(img)[0] * np.shape(img)[1]
            # Calculate the snr,backdigma is varriance
            print( (totalflux - backtotal) / (backsigma) ** 0.5)

    # -----------------------------------------------------------------------------------------------
    # Function defintion the following function is used to calculate the Ellipicity under KSB Model
    # Clip PSF is used to cut part of psf from the orginal psf
    def clippsf(self, orgpsf, psfsize=None):
        # clip psf according to its maximal value to a predefined size
        # psfsize should be odd number
        # orgpsf is the original psf 2D array
        temcoor = np.where(orgpsf == np.max(orgpsf))
        # print temcoor
        cooy = temcoor[0][0]
        coox = temcoor[1][0]
        if psfsize is not None:
            halfsize = psfsize / 2
            newpsf = orgpsf[cooy - halfsize - 1:cooy + halfsize, coox - halfsize - 1:coox + halfsize]
        else:
            halfsize = np.shape(orgpsf)[0]
            maxsize = np.min([cooy, coox, halfsize])
            newpsf = orgpsf[cooy - maxsize:cooy + maxsize + 1, coox - maxsize:coox + maxsize + 1]
        return newpsf

    # Storexy is used to store the x and y distribution matrix nx and ny should be the size of matrix in x and y
    def storeXY(self, nx, ny, r = 9):
        XX = np.zeros((nx, ny))
        XY = np.zeros((nx, ny))
        YY = np.zeros((nx, ny))
        for i in range(0, nx):
            x = 0.5 + i - (nx) / 2.0
            for j in range(0, ny):
                y = 0.5 + j - (ny) / 2.0
                if (r < 10):
                    XX[i, j] = x * x
                    XY[i, j] = x * y
                    YY[i, j] = y * y
        return XX, XY, YY

    # get Quad is used to calculate the quadic of the matrix
    def getQuad(self, img, XX, XY, YY, mod = 0):  # returns the (unweighted) quadrupole matrix of an (nx x ny) img
        # img is the orginal image with size of N M
        # XX XY and YY are the 2D matrix calculated by storeXY
        quad = np.array([[0.0, 0.0], [0.0, 0.0]])
        quad[1][0] = (img.dot(XY)).sum()
        quad[0][0] = (img.dot(YY)).sum() - mod  ##### WARNING: X AND Y HAVE BEEN SWAPPED TO
        quad[1][1] = (img.dot(XX)).sum() - mod  ##### ACCOUNT FOR NUMPY BEING (Y,X)
        quad[0][1] = quad[1][0]
        return quad

    # Used to calculate the quad parameter in KSB
    def polE(self, quad):  # returns the KSB "polarization parameters" defined in KSB Eq 3.2
        e = np.array([[0.0], [0.0]])
        q1 = quad[0][0] - quad[1][1]
        q2 = 2.0 * quad[1][0]
        T = (quad[0][0] + quad[1][1]) / 2.
        T += 2. * np.sqrt(np.linalg.det(quad)) / 2.
        e[0] = q1 / T
        e[1] = q2 / T
        return e

    # Main function used to calculate the Ellipisicity
    def get_ellipisicity(self):
        # need to install package of skimage
        from skimage.measure import regionprops
        # orgpsf=clippsf(orgpsf)
        for fit in self.fitfile:
            hdul = fits.open(fit)
            img = hdul[0].data
            orgpsf = self.clippsf(img)
            sizey, sizex = np.shape(orgpsf)
        #    if(sizey != sizex):
        #        return 0,0
            XX, XY, YY = self.storeXY(sizex, sizey)
            quad = self.getQuad(orgpsf, XX, XY, YY)
            ellipse1, ellipse2 = self.polE(quad)
            print(ellipse1[0], ellipse2[0])
