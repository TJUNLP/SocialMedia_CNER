# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-


import codecs


def evaluation_NER(testresult, moretag_list=None):

    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.

    # total_predict_right_nom = 0.
    # total_predict_nom = 0.
    # total_right_nom = 0.
    #
    # total_predict_right_nam = 0.
    # total_predict_nam = 0.
    # total_right_nam = 0.

    right4entlensDict = {1: 0, 2: 0, 3: 0, 4: 0}
    predict4entlensDict = {1: 0, 2: 0, 3: 0, 4: 0}
    AllrightDict = {1: 0, 2: 0, 3: 0, 4: 0}

    Allright4senlenDict = {}
    right4senlenDict = {}
    predict4senlenDict = {}
    for senlen in range(1, 176):
        Allright4senlenDict[senlen//10 * 10] = 0.
        right4senlenDict[senlen//10 * 10] = 0.
        predict4senlenDict[senlen//10 * 10] = 0.

    for id,sent in enumerate(testresult):
        ptag = sent[0]
        ttag = sent[1]
        # print('ptag--'+str(ptag))
        # print('ttag--'+str(ttag))
        tlist = ['LOC', 'ORG', 'PER', 'GPE']

        i = 0
        while i < len(ttag):
            # print(id)
            # print('ttag['+str(i)+'] is-'+ttag[i]+'-')

            if ttag[i] == '' or ttag[i] == 'O' or ttag[i] == 'N':
                i += 1
                continue

            for type in tlist:
                if ttag[i].__contains__('S-'+type):
                    total_right += 1.
                    # if '.NAM' in moretag_list[id][i]:
                    #     total_right_nam += 1.
                    # elif '.NOM' in moretag_list[id][i]:
                    #     total_right_nom += 1.

                    AllrightDict[1] += 1.
                    Allright4senlenDict[len(ttag) // 10 * 10] += 1.
                    break

                elif ttag[i].__contains__('B-'+type):
                    j = i + 1
                    while j < len(ttag):
                        if ttag[j].__contains__('I-'+type):
                            j += 1
                        elif ttag[j].__contains__('E-'+type):
                            total_right += 1.
                            # if '.NAM' in moretag_list[id][i]:
                            #     total_right_nam += 1.
                            # elif '.NOM' in moretag_list[id][i]:
                            #     total_right_nom += 1.

                            lens = min(j - i + 1, 4)
                            AllrightDict[lens] += 1.
                            Allright4senlenDict[len(ttag) // 10 * 10] += 1.

                            i = j
                            break
                        else:
                            print('error-'+type, i)
                            i = j-1
                            break
                    break

                elif ttag[i].__contains__('I-') or ttag[i].__contains__('E-'):
                    print('error-other', id, '  --' + ttag[i] + '--')
                    print(ttag)
            i += 1


        # print('total_right = ', total_right)

        i = 0
        while i < len(ptag):

            for type in tlist:

                if ptag[i] == '' or ptag[i] == 'O' or ptag[i] == 'N':
                    break

                elif ptag[i].__contains__('S-'+type):
                    total_predict += 1.
                    # if '.NAM' in moretag_list[id][i]:
                    #     total_predict_nam += 1.
                    # elif '.NOM' in moretag_list[id][i]:
                    #     total_predict_nom += 1.

                    predict4entlensDict[1] += 1.
                    predict4senlenDict[len(ttag) // 10 * 10] += 1.
                    if ttag[i].__contains__('S-'+type):
                        total_predict_right += 1.
                        # if '.NAM' in moretag_list[id][i]:
                        #     total_predict_right_nam += 1.
                        # elif '.NOM' in moretag_list[id][i]:
                        #     total_predict_right_nom += 1.
                        right4entlensDict[1] += 1.
                        right4senlenDict[len(ttag) // 10 * 10] += 1.

                    break

                elif ptag[i].__contains__('B-'+type):

                    j = i + 1
                    if j == len(ptag):
                        break

                    while j < len(ptag):
                        if ptag[j].__contains__('I-'+type):
                            j += 1
                        elif ptag[j].__contains__('E-'+type):
                            total_predict += 1
                            # if '.NAM' in moretag_list[id][i]:
                            #     total_predict_nam += 1.
                            # elif '.NOM' in moretag_list[id][i]:
                            #     total_predict_nom += 1.

                            predict4entlensDict[min(j-i+1, 4)] += 1.
                            predict4senlenDict[len(ttag) // 10 * 10] += 1.
                            if ttag[i].__contains__('B-'+type) and ttag[j].__contains__('E-'+type):
                                total_predict_right += 1
                                # if '.NAM' in moretag_list[id][i]:
                                #     total_predict_right_nam += 1.
                                # elif '.NOM' in moretag_list[id][i]:
                                #     total_predict_right_nom += 1.
                                lens = min(j-i+1, 4)
                                right4entlensDict[lens] += 1.
                                right4senlenDict[len(ttag) // 10 * 10] += 1.
                            i = j
                            break
                        else:
                            i = j-1
                            break
                    break

            i += 1
        # print('total_predict_right = ', total_predict_right)
        # print('total_predict = ', total_predict)

    P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R) if P != 0 else 0

    return P, R, F, total_predict_right, total_predict, total_right



