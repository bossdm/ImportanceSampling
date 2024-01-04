def print_MSE_rows(MSEList,writefile):
    SortedMSEList = sorted(MSEList)
    best_MSE = SortedMSEList[0]
    for index, MSE in enumerate(MSEList):
        r = SortedMSEList.index(MSE)
        if r == 0:
            writefile.write(r"& \underline{\underline{\textbf{%.4f}}} " % (MSEList[index]))
        elif MSEList[index] == best_MSE:  # underline all ties
            writefile.write(r"& \underline{\underline{\textbf{%.4f}}} " % (MSEList[index]))
        elif r == 1:  # second performance (in case no tie)
            writefile.write(r"& \underline{\textbf{%.4f}} " % (MSEList[index]))
        elif r == 2:  # third performance (in case no tie)
            writefile.write(r"& \textbf{%.4f} " % (MSEList[index]))
        else:
            writefile.write(r"& %.4f " % (MSEList[index]))

def print_MSE_rows_precision(MSEList,writefile):
    SortedMSEList = sorted(MSEList)
    best_MSE = SortedMSEList[0]
    for index, MSE in enumerate(MSEList):
        r = SortedMSEList.index(MSE)
        if r == 0:
            writefile.write(r"& \underline{\underline{\textbf{%.5f}}} " % (MSEList[index]))
        elif MSEList[index] == best_MSE:  # underline all ties
            writefile.write(r"& \underline{\underline{\textbf{%.5f}}} " % (MSEList[index]))
        elif r == 1:  # second performance (in case no tie)
            writefile.write(r"& \underline{\textbf{%.5f}} " % (MSEList[index]))
        elif r == 2:  # third performance (in case no tie)
            writefile.write(r"& \textbf{%.5f} " % (MSEList[index]))
        else:
            writefile.write(r"& %.5f " % (MSEList[index]))