#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
import os
import torch
import numpy as np
from shapely.geometry import Polygon as plg
from model import EAST
from detect import detect_boxes


def evaluate_method(all_gt_boxes, all_pred_boxes):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """
    """
    for module,alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module)
    """
    def polygon_from_points(points):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """
        resBoxes = np.empty([1, 8], dtype='int32')
        resBoxes[0, 0] = int(points[0])
        resBoxes[0, 4] = int(points[1])
        resBoxes[0, 1] = int(points[2])
        resBoxes[0, 5] = int(points[3])
        resBoxes[0, 2] = int(points[4])
        resBoxes[0, 6] = int(points[5])
        resBoxes[0, 3] = int(points[6])
        resBoxes[0, 7] = int(points[7])
        pointMat = resBoxes[0].reshape([2, 4]).T
        return plg(pointMat)

    def get_union(pD, pG):
        areaA = pD.area
        areaB = pG.area
        return areaA + areaB - get_intersection(pD, pG)

    def get_intersection_over_union(pD, pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG)
        except:
            return 0

    def get_intersection(pD, pG):
        pInt = pD & pG
        try:
            if len(pInt) == 0:
                return 0
        except:
            return pInt.area

    matchedSum = 0

    numGlobalCareGt = 0
    numGlobalCareDet = 0

    for box_id in range(len(all_gt_boxes)):
        gt_boxes = all_gt_boxes[box_id]

        recall = 0
        precision = 0
        hmean = 0

        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        gtDontCarePolsNum = []
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        evaluationLog = ""

        pointsList = [gt_box[:-1] for gt_box in gt_boxes]
        transcriptionsList = [gt_box[-1] for gt_box in gt_boxes]

        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"

            gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtPolPoints.append(points)

            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        evaluationLog += "GT polygons: " + str(len(gtPols)) + (" (" + str(len(gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum) > 0 else "\n")

        if True:
            pred_boxes = all_pred_boxes[box_id]

            pointsList = [pred_box[:-1] for pred_box in pred_boxes]
            for n in range(len(pointsList)):
                points = pointsList[n]

                detPol = polygon_from_points(points)
                detPols.append(detPol)
                detPolPoints.append(points)
                if len(gtDontCarePolsNum) > 0:
                    for dontCarePol in gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePol]
                        intersected_area = get_intersection(dontCarePol, detPol)
                        pdDimensions = detPol.area
                        precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                        if (precision > 0.5):
                            detDontCarePolsNum.append(len(detPols) - 1)
                            break

            evaluationLog += "DET polygons: " + str(len(detPols)) + (" (" + str(len(detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum) > 0 else "\n")

            if len(gtPols) > 0 and len(detPols) > 0:
                outputShape = [len(gtPols), len(detPols)]
                iouMat = np.empty(outputShape)
                gtRectMat = np.zeros(len(gtPols), np.int8)
                detRectMat = np.zeros(len(detPols), np.int8)
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]
                        iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

                for gtNum in range(len(gtPols)):
                    match = False
                    for detNum in range(len(detPols)):
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                            if iouMat[gtNum, detNum] > 0.5:
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                detMatched += 1
                                pairs.append({'gt': gtNum, 'det': detNum})
                                detMatchedNums.append(detNum)
                                evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"
                                match = True

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare

        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
        print('=='*28)

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)

    methodMetrics = {'precision': methodPrecision,
                     'recall': methodRecall,
                     'hmean': methodHmean}

    resDict = {'calculated': True, 'method': methodMetrics}

    return resDict


def read_gt_boxes(test_gt_path):
    gt_files = os.listdir(test_gt_path)
    gt_files = sorted([os.path.join(test_gt_path, gt_file)
                       for gt_file in gt_files])

    gt_boxes = list()

    for i, gt_file in enumerate(gt_files):
        print("evaluating {} image".format(i), end='\r')
        gt_text = open(gt_file, 'r').read().split('\n')[:-1]
        boxes = list()

        for gt_line in gt_text:
            boxes.append(gt_line.split(','))
            gt_boxes.append(boxes)

        return gt_boxes


def compute_hmean(model, device):
    print('EAST <==> Evaluation <==> Compute Hmean <==> Begin')

    test_img_path = '../ICDAR_2015/test_img/'
    test_gt_path = '../ICDAR_2015/test_gt/'

    print("Predicting boxes...")
    pred_boxes = detect_boxes(model, device, test_img_path)

    print("Reading GT boxes...")
    gt_boxes = read_gt_boxes(test_gt_path)

    print("Evaluating Result...")
    resDict = evaluate_method(gt_boxes, pred_boxes)

    recall = resDict['method']['recall']
    precision = resDict['method']['precision']
    hmean = resDict['method']['hmean']

    print("F1-Score: {}".format(hmean))

    return hmean


if __name__ == "__main__":
    model_path = './pths/east.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST().to(device)
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    hmean = compute_hmean(model, device)

    print("F1-Score: {}".format(hmean))
