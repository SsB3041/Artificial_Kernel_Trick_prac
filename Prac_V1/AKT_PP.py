from re import S
import pandas as pd
import numpy as np
import math
from scipy.spatial import distance
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
import pickle
from sklearn.manifold import TSNE
warnings.filterwarnings("ignore")
sc= int(input())

# input_name = input("file name = ")

if sc == 1 :
    print("Standard gogo")
    print('csv file name = ')
    f_name = input()
    print("result뒤에 붙일말은 생각하셨나요?")
    times_name = "test_1_evaluation"
    file_name = input()
    # dk= pd.read_csv("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/{}_Test.txt".format(f_name),sep="\t",engine="python",encoding="cp949")
    # data = pd.read_csv("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/{}_Test.txt".format(f_name),sep="\t",engine="python",encoding="cp949")
    # data = pd.read_csv("E:/공간이없두/dataSpread/{}_Test.txt".format(f_name),sep="\t",engine="python",encoding="cp949")
    dk = pd.read_csv("C:/Users/82108/Desktop/code/Pos/공간이없두 (2)/after0202/New spread/dataset_aftermeeting/total_tsne_file.csv")
    # dk = pd.read_csv("E:/공간이없두/dataSpread/{}_Test.csv".format(f_name))
    ## dummy 제작     
    
    dk['dummy']= 1    
    dk['dummy3']= 0.1
    dk['dummy4']= 0.01    
    
    C_D=dk.loc[:,"C"].mul(dk.loc[:,"C"])
    C_D=pd.DataFrame({"C_D":C_D})
    Mn_under = dk.loc[:,"Mn"].mul(dk.loc[:,"dummy4"])
    Mn_under = pd.DataFrame({"Mn_under":Mn_under})
    Cr_under = dk.loc[:,"Cr"].mul(dk.loc[:,"dummy4"])
    Cr_under = pd.DataFrame({"Cr_under":Cr_under})
    Ni_under = dk.loc[:,"Ni"].mul(dk.loc[:,"dummy4"])
    Ni_under = pd.DataFrame({"Ni_under":Ni_under})
    Mo_under = dk.loc[:,"Mo"].mul(dk.loc[:,"dummy4"])
    Mo_under = pd.DataFrame({"Mo_under":Mo_under})   
     
    C_B = dk.loc[:,"C"].mul(dk.loc[:,"B"])
    C_B = pd.DataFrame({"C_B":C_B})
    C_Ni = dk.loc[:,"C"].mul(dk.loc[:,"Ni"])
    C_Ni = pd.DataFrame({"C_Ni":C_Ni})
    Mo_Ni = dk.loc[:,"Mo"].mul(dk.loc[:,"Ni"])
    Mo_Ni = pd.DataFrame({"Mo_Ni":Mo_Ni})
    V_Ni = dk.loc[:,"V"].mul(dk.loc[:,"Ni"])
    V_Ni = pd.DataFrame({"V_Ni":V_Ni})
    Mn_Ni = dk.loc[:,"Mn"].mul(dk.loc[:,"Ni"])
    Mn_Ni = pd.DataFrame({"Mn_Ni":Mn_Ni})
    Cr_Mn = dk.loc[:,"Cr"].mul(dk.loc[:,"Mn"])
    Cr_Mn = pd.DataFrame({"Cr_Mn":Cr_Mn})
    
    
    
    B_up = dk.loc[:,"dummy"].div(dk.loc[:,"B"])
    B_up = pd.DataFrame({"B_up":B_up})
    V_up = dk.loc[:,"dummy"].div(dk.loc[:,"V"])
    V_up = pd.DataFrame({"V_up":V_up})
    Cu_up = dk.loc[:,"dummy"].div(dk.loc[:,"Cu"])
    Cu_up = pd.DataFrame({"Cu_up":Cu_up})
    Si_up = dk.loc[:,"dummy"].div(dk.loc[:,"Si"])
    Si_up = pd.DataFrame({"Si_up":Si_up})
    Al_up = dk.loc[:,"dummy"].div(dk.loc[:,"Al"])
    Al_up = pd.DataFrame({"Al_up":Al_up})
    Ti_up = dk.loc[:,"dummy"].div(dk.loc[:,"Ti"])
    Ti_up = pd.DataFrame({"Ti_up":Ti_up})
    Pb_up = dk.loc[:,"dummy"].div(dk.loc[:,"Pb"])
    Pb_up = pd.DataFrame({"Pb_up":Pb_up})
    W_up = dk.loc[:,"dummy"].div(dk.loc[:,"W"])
    W_up = pd.DataFrame({"W_up":W_up})
    
    
    
    
    ##################### 0120 추가
    C_Mn = dk.loc[:,"C"].mul(dk.loc[:,"Mn"])
    C_Mn = pd.DataFrame({"C_Mn":C_Mn})
    B_Mn = dk.loc[:,"B"].mul(dk.loc[:,"Mn"])
    B_Mn = pd.DataFrame({"B_Mn":B_Mn})
    
    
    
    dk_sub=pd.concat([dk,Mn_under],axis=1)
    Mn_C = dk_sub.loc[:,"Mn_under"].mul(dk_sub.loc[:,"C"])
    Mn_C = pd.DataFrame({"Mn_C":Mn_C})
    
    ######################################
    # AL TI W PB
    dk_name = dk.drop(["dummy","dummy3","dummy4"],axis=1)
    
    
    dk_under=pd.concat([C_D,Mn_under,Cr_under,C_B,C_Ni,Mo_Ni,V_Ni,Ni_under,Mn_Ni,Cr_Mn,Mo_under],axis=1)    
    dk_up=pd.concat([B_up,V_up,Si_up,Al_up,Ti_up,Pb_up,W_up],axis=1)
    
    
    dk_under["dummy"] = 1
    C_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"C_D"])
    C_up = pd.DataFrame({"C_up":C_up})
    Mn_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Mn_under"])
    Mn_up = pd.DataFrame({"Mn_up":Mn_up})
    Cr_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Cr_under"])
    Cr_up = pd.DataFrame({"Cr_up":Cr_up})
    Ni_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Ni_under"])
    Ni_up = pd.DataFrame({"Ni_up":Ni_up})
    Mo_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Mo_under"])
    Mo_up = pd.DataFrame({"Mo_up":Mo_up})    
    C_B_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"C_B"])
    C_B_up = pd.DataFrame({"C_B_up":C_B_up})
    C_Ni_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"C_Ni"])
    C_Ni_up = pd.DataFrame({"C_Ni_up":C_Ni_up})
    Mo_Ni_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Mo_Ni"])
    Mo_Ni_up = pd.DataFrame({"Mo_Ni_up":Mo_Ni_up})
    V_Ni_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"V_Ni"])
    V_Ni_up = pd.DataFrame({"V_Ni_up":V_Ni_up})
    Mn_Ni_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Mn_Ni"])
    Mn_Ni_up = pd.DataFrame({"Mn_Ni_up":Mn_Ni_up})
    Cr_Mn_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Cr_Mn"])
    Cr_Mn_up = pd.DataFrame({"Cr_Mn_up":Cr_Mn_up})
    
    
    ### C_Mn ==  걍 곱하기 , Mn_C는 0.1배
    # C_Mn_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"C_Mn"])
    # C_Mn_up = pd.DataFrame({"C_Mn_up":C_Mn_up})
    # B_Mn_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"B_Mn"])
    # B_Mn_up = pd.DataFrame({"B_Mn_up":B_Mn_up})
    # Mn_C_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Mn_C"])
    # Mn_C_up = pd.DataFrame({"Mn_C_up":Mn_C_up})
    
    
    
    
    # dk_under_to_up = pd.concat([C_up,Mn_up,Cr_up,C_B_up,C_Ni_up,Mo_Ni_up,V_Ni_up],axis=1)
    dk_under_to_up = pd.concat([C_up,Mn_up,Cr_up,Mo_up],axis=1)
    
    final=pd.concat([dk_name,dk_under_to_up,dk_up],axis=1)
    final=pd.DataFrame(final)
    
     
    
    
    
    final=final.replace([np.inf,-np.inf],200000)
    data=final.fillna(0)

    data.to_csv("C:/Users/82108/Desktop/code/Pos/공간이없두 (2)/nano/{}_kernel_on.txt".format(f_name),sep="\t",index=False)
    # # ###### print(data)
    data=pd.read_csv("C:/Users/82108/Desktop/code/Pos/공간이없두 (2)/nano/{}_kernel_on.txt".format(f_name),sep="\t",engine="python",encoding="cp949")
    
    X=data.loc[:,"C":]
    Y=data.iloc[:,0]
    # Y2=data.iloc[:,1]
    SS= StandardScaler()
    X_scaled= SS.fit_transform(X)
    pca=PCA(n_components=2)
    pca_result=pca.fit_transform(X_scaled)
    # pca_result=pca.fit_transform(X)
    print(pca.explained_variance_ratio_)
    
    with open("C:/Users/82108/Desktop/code/Pos/공간이없두 (2)/nano/dummyfile_pikle2.pkl","wb") as f :
        pickle.dump(pca,f,pickle.HIGHEST_PROTOCOL)
    pca_pd = pd.DataFrame(pca_result,columns=["PCA1","PCA2"])
    pca_pd['label'] = Y
    # pca_pd["class"] = Y2
    pca_pd.to_csv("C:/Users/82108/Desktop/code/Pos/공간이없두 (2)/nano/dummyfile2.csv",index=False)
    pca_pd.to_csv("C:/Users/82108/Desktop/code/Pos/공간이없두 (2)/nano/dummyfile2.txt",sep="\t",index=False)
    cumsum_SS = np.cumsum(pca.explained_variance_ratio_)
    SS_Di = np.argmax(cumsum_SS >=0.95) +1
    print("SS 차원의 수 :" ,SS_Di)
    
    
    
    
    ##### evaluate
          
    base= pickle.load(open("C:/Users/82108/Desktop/code/Pos/공간이없두 (2)/nano/dummyfile_pikle2.pkl",'rb'))
    result = pd.read_csv("C:/Users/82108/Desktop/code/Pos/공간이없두 (2)/nano/dummyfile2.csv")
    
    # input_file=pd.read_csv("E:/공간이없두/JASO/input_file2_{}_{}_{}.csv".format(f_name,times_name,file_name))
    print("input_file넣어주세요")
    test_file_num=input()
    input_file=pd.read_csv("C:/Users/82108/Desktop/code/Pos/공간이없두 (2)/random/Total/test_1_Total.csv")
    # input_file=pd.read_csv("E:/공간이없두/after 01 20/하압!/test_1_{}.txt".format(test_file_num),sep="\t",engine="python",encoding="cp949")
    input_file=input_file.fillna(0) 
    input_file=input_file.iloc[:,:-1]

    input_file['dummy']= 1    
    input_file['dummy3']= 0.1    
    input_file['dummy4']= 0.01 
    
     
    C_D=input_file.loc[:,"C"].mul(input_file.loc[:,"C"])
    C_D=pd.DataFrame({"C_D":C_D})
    Mn_under = input_file.loc[:,"Mn"].mul(input_file.loc[:,"dummy4"])
    Mn_under = pd.DataFrame({"Mn_under":Mn_under})
    Cr_under = input_file.loc[:,"Cr"].mul(input_file.loc[:,"dummy4"])
    Cr_under = pd.DataFrame({"Cr_under":Cr_under})
    Ni_under = input_file.loc[:,"Ni"].mul(input_file.loc[:,"dummy4"])
    Ni_under = pd.DataFrame({"Ni_under":Ni_under})
    Mo_under = input_file.loc[:,"Mo"].mul(input_file.loc[:,"dummy4"])
    Mo_under = pd.DataFrame({"Mo_under":Mo_under})
    
    C_B = input_file.loc[:,"C"].mul(input_file.loc[:,"B"])
    C_B = pd.DataFrame({"C_B":C_B})
    C_Ni = input_file.loc[:,"C"].mul(input_file.loc[:,"Ni"])
    C_Ni = pd.DataFrame({"C_Ni":C_Ni})
    Mo_Ni = input_file.loc[:,"Mo"].mul(input_file.loc[:,"Ni"])
    Mo_Ni = pd.DataFrame({"Mo_Ni":Mo_Ni})
    V_Ni = input_file.loc[:,"V"].mul(input_file.loc[:,"Ni"])
    V_Ni = pd.DataFrame({"V_Ni":V_Ni})
    Mn_Ni = input_file.loc[:,"Mn"].mul(input_file.loc[:,"Ni"])
    Mn_Ni = pd.DataFrame({"Mn_Ni":Mn_Ni})
    Cr_Mn = input_file.loc[:,"Cr"].mul(input_file.loc[:,"Mn"])
    Cr_Mn = pd.DataFrame({"Cr_Mn":Cr_Mn})
    
    
    
    B_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"B"])
    B_up = pd.DataFrame({"B_up":B_up})
    V_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"V"])
    V_up = pd.DataFrame({"V_up":V_up})
    Cu_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"Cu"])
    Cu_up = pd.DataFrame({"Cu_up":Cu_up})
    Si_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"Si"])
    Si_up = pd.DataFrame({"Si_up":Si_up})
    Al_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"Al"])
    Al_up = pd.DataFrame({"Al_up":Al_up})
    Ti_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"Ti"])
    Ti_up = pd.DataFrame({"Ti_up":Ti_up})
    Pb_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"Pb"])
    Pb_up = pd.DataFrame({"Pb_up":Pb_up})
    W_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"W"])
    W_up = pd.DataFrame({"W_up":W_up})
    
    
    ###### 01 20 
    C_Mn = input_file.loc[:,"C"].mul(input_file.loc[:,"Mn"])
    C_Mn = pd.DataFrame({"C_Mn":C_Mn})
    B_Mn = input_file.loc[:,"B"].mul(input_file.loc[:,"Mn"])
    B_Mn = pd.DataFrame({"B_Mn":B_Mn})
    
    
    
    ######
    input_file_sub=pd.concat([input_file,Mn_under],axis=1)
    Mn_C = input_file_sub.loc[:,"Mn_under"].mul(input_file_sub.loc[:,"C"])
    Mn_C = pd.DataFrame({"Mn_C":Mn_C})
       
       
    input_file_name  = input_file.drop(["dummy","dummy3","dummy4"],axis=1)
    
    
    input_file_under=pd.concat([C_D,Mn_under,Cr_under,C_B,C_Ni,Mo_Ni,V_Ni,Ni_under,Mn_Ni,Cr_Mn,Mo_under],axis=1)    
    input_file_up=pd.concat([B_up,V_up,Si_up,Al_up,Ti_up,Pb_up,W_up],axis=1)
    
    
    input_file_under["dummy"] = 1
    C_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"C_D"])
    C_up = pd.DataFrame({"C_up":C_up})
    Mn_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Mn_under"])
    Mn_up = pd.DataFrame({"Mn_up":Mn_up})
    Cr_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Cr_under"])
    Cr_up = pd.DataFrame({"Cr_up":Cr_up})
    Ni_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Ni_under"])
    Ni_up = pd.DataFrame({"Ni_up":Ni_up})
    Mo_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Mo_under"])
    Mo_up = pd.DataFrame({"Mo_up":Mo_up})    
    C_B_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"C_B"])
    C_B_up = pd.DataFrame({"C_B_up":C_B_up})
    C_Ni_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"C_Ni"])
    C_Ni_up = pd.DataFrame({"C_Ni_up":C_Ni_up})
    Mo_Ni_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Mo_Ni"])
    Mo_Ni_up = pd.DataFrame({"Mo_Ni_up":Mo_Ni_up})
    V_Ni_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"V_Ni"])
    V_Ni_up = pd.DataFrame({"V_Ni_up":V_Ni_up})
    Mn_Ni_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Mn_Ni"])
    Mn_Ni_up = pd.DataFrame({"Mn_Ni_up":Mn_Ni_up})
    Cr_Mn_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Cr_Mn"])
    Cr_Mn_up = pd.DataFrame({"Cr_Mn_up":Cr_Mn_up})
    
    
    
    ### C_Mn ==  걍 곱하기 , Mn_C는 0.1배
    # C_Mn_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"C_Mn"])
    # C_Mn_up = pd.DataFrame({"C_Mn_up":C_Mn_up})
    # B_Mn_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"B_Mn"])
    # B_Mn_up = pd.DataFrame({"B_Mn_up":B_Mn_up})
    # Mn_C_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Mn_C"])
    # Mn_C_up = pd.DataFrame({"Mn_C_up":Mn_C_up})
    
    
    
    
    # input_file_under_to_up = pd.concat([C_up,Mn_up,Cr_up,C_B_up,C_Ni_up,Mo_Ni_up,V_Ni_up],axis=1)
    input_file_under_to_up = pd.concat([C_up,Mn_up,Cr_up,Mo_up],axis=1)
    
    final=pd.concat([input_file_name,input_file_under_to_up,input_file_up],axis=1)
    final=pd.DataFrame(final)
    # print(final)
    final=final.replace([np.inf,-np.inf],200000)
    
    ## NK 일때 열긔
    # final=input_file.loc[:,:"N"]
    
    
    
    final=SS.transform(final)


    pca_target = base.transform(final)
    
    dt1=[]
    print(type(dt1))
    dt2=[]
    dt3=[]
    dt4=[]
    dt5=[]
    dt6=[]
    dt7=[]
    dt8=[]
    dt9=[]
    dt10=[]
    
    
    for k in range(input_file.shape[0]):       
        target1 = pca_target [k][0]
        target2 = pca_target [k][1]
        print(target1,target2) 
        final=pd.DataFrame()
        
        def euclidean_distance(target1,target2,standard1,standard2) :
            distance=0
            distance1=((target1-standard1)) ** 2
            distance2=((target2-standard2)) ** 2
            distance += (distance1+distance2)      
            return distance **0.5
        lists=pd.DataFrame()
        result=result.groupby(["label"],as_index=False).mean()
        # sns.lmplot("PCA1", "PCA2", data=pca_pd , fit_reg=False, scatter_kws={"s":4}, hue="label")
        # plt.show()


        for i in range(0,result.shape[0]) :
            # print(i+'/'+result.shape[0]-1)
            print(i,'/',result.shape[0])
            name=result.loc[i,"label"]
            standard1=result.loc[i,"PCA1"]
            standard2=result.loc[i,"PCA2"]           
            geori=pd.DataFrame([{"label":name,"distance":euclidean_distance(target1,target2,standard1,standard2)}])
            lists=lists.append(geori)
        final=final.append(lists)

        dk=final.sort_values(by="distance")
        # dk.to_csv("GB_NEW_NOT_scale_with__input_distance.csv")
        dt=dk.iloc[:10,:]
        
        dt1.append(dt.iloc[0,0])
        dt2.append(dt.iloc[1,0])
        dt3.append(dt.iloc[2,0])
        dt4.append(dt.iloc[3,0])
        dt5.append(dt.iloc[4,0])
        dt6.append(dt.iloc[5,0])
        dt7.append(dt.iloc[6,0])
        dt8.append(dt.iloc[7,0])
        dt9.append(dt.iloc[8,0])
        dt10.append(dt.iloc[9,0])
        
        
        # dt=pd.concat([dt,input_file],axis=0)
        # dt.to_csv("E:/공간이없두/JASO/{}_NEW_Standard_scale_with__input_distance_reversion_{}_{}.csv".format(f_name,times_name,file_name),index=False)
    
    input_file.insert(0,"rank1",dt1)
    input_file.insert(0,"rank2",dt2)
    input_file.insert(0,"rank3",dt3)
    input_file.insert(0,"rank4",dt4)
    input_file.insert(0,"rank5",dt5)
    input_file.insert(0,"rank6",dt6)
    input_file.insert(0,"rank7",dt7)
    input_file.insert(0,"rank8",dt8)
    input_file.insert(0,"rank9",dt9)
    input_file.insert(0,"rank10",dt10)
    
    input_file.to_csv("C:/Users/82108/Desktop/code/Pos/공간이없두 (2)/nano/잠깐_result2.csv",index=False)
    
    
    
if sc == 2 :
    print("Standard gogo")
    print('csv file name = ')
    f_name = "GB"
    # print("몇번째시도인가요?")
    # times_name= "ASCM320H" 
    times_name = "test_1_evaluation"
    # file_name = input()
    dk= pd.read_csv("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/{}_Test.txt".format(f_name),sep="\t",engine="python",encoding="cp949")
    # data = pd.read_csv("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/{}_Test.txt".format(f_name),sep="\t",engine="python",encoding="cp949")
    ## dummy 제작     
    
    dk['dummy']= 1    
    dk['dummy3']= 0.1    
    
    C_D=dk.loc[:,"C"].mul(dk.loc[:,"C"])
    C_D=pd.DataFrame({"C_D":C_D})
    Mn_under = dk.loc[:,"Mn"].mul(dk.loc[:,"dummy"])
    Mn_under = pd.DataFrame({"Mn_under":Mn_under})
    Cr_under = dk.loc[:,"Cr"].mul(dk.loc[:,"dummy"])
    Cr_under = pd.DataFrame({"Cr_under":Cr_under})
    C_B = dk.loc[:,"C"].mul(dk.loc[:,"B"])
    C_B = pd.DataFrame({"C_B":C_B})
    C_Ni = dk.loc[:,"C"].mul(dk.loc[:,"Ni"])
    C_Ni = pd.DataFrame({"C_Ni":C_Ni})
    Mo_Ni = dk.loc[:,"Mo"].mul(dk.loc[:,"Ni"])
    Mo_Ni = pd.DataFrame({"Mo_Ni":Mo_Ni})
    V_Ni = dk.loc[:,"V"].mul(dk.loc[:,"Ni"])
    V_Ni = pd.DataFrame({"V_Ni":V_Ni})    
    B_up = dk.loc[:,"dummy"].div(dk.loc[:,"B"])
    B_up = pd.DataFrame({"B_up":B_up})
    V_up = dk.loc[:,"dummy"].div(dk.loc[:,"V"])
    V_up = pd.DataFrame({"V_up":V_up})
    Cu_up = dk.loc[:,"dummy"].div(dk.loc[:,"Cu"])
    Cu_up = pd.DataFrame({"Cu_up":Cu_up})
    Si_up = dk.loc[:,"dummy"].div(dk.loc[:,"Si"])
    Si_up = pd.DataFrame({"Si_up":Si_up})
    ##################### 0120 추가
    C_Mn = dk.loc[:,"C"].mul(dk.loc[:,"Mn"])
    C_Mn = pd.DataFrame({"C_Mn":C_Mn})
    B_Mn = dk.loc[:,"B"].mul(dk.loc[:,"Mn"])
    B_Mn = pd.DataFrame({"B_Mn":B_Mn})
    
    dk_sub=pd.concat([dk,Mn_under],axis=1)
    Mn_C = dk_sub.loc[:,"Mn_under"].mul(dk_sub.loc[:,"C"])
    Mn_C = pd.DataFrame({"Mn_C":Mn_C})
    
    ######################################
    
    dk_name = dk.drop(["dummy","dummy3","Pb","N","Nb"],axis=1)
    
    
    dk_under=pd.concat([C_D,Mn_under,Cr_under,C_B,C_Ni,Mo_Ni,V_Ni,C_Mn,B_Mn,Mn_C],axis=1)    
    dk_up=pd.concat([B_up,V_up,Cu_up,Si_up],axis=1)
    
    
    dk_under["dummy"] = 1
    C_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"C_D"])
    C_up = pd.DataFrame({"C_up":C_up})
    Mn_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Mn_under"])
    Mn_up = pd.DataFrame({"Mn_up":Mn_up})
    Cr_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Cr_under"])
    Cr_up = pd.DataFrame({"Cr_up":Cr_up})
    C_B_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"C_B"])
    C_B_up = pd.DataFrame({"C_B_up":C_B_up})
    C_Ni_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"C_Ni"])
    C_Ni_up = pd.DataFrame({"C_Ni_up":C_Ni_up})
    Mo_Ni_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Mo_Ni"])
    Mo_Ni_up = pd.DataFrame({"Mo_Ni_up":Mo_Ni_up})
    V_Ni_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"V_Ni"])
    V_Ni_up = pd.DataFrame({"V_Ni_up":V_Ni_up})
    ### C_Mn ==  걍 곱하기 , Mn_C는 0.1배
    C_Mn_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"C_Mn"])
    C_Mn_up = pd.DataFrame({"C_Mn_up":C_Mn_up})
    B_Mn_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"B_Mn"])
    B_Mn_up = pd.DataFrame({"B_Mn_up":B_Mn_up})
    Mn_C_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Mn_C"])
    Mn_C_up = pd.DataFrame({"Mn_C_up":Mn_C_up})
    
    
    
    
    # dk_under_to_up = pd.concat([C_up,Mn_up,Cr_up,C_B_up,C_Ni_up,Mo_Ni_up,V_Ni_up],axis=1)
    dk_under_to_up = pd.concat([C_up,Mn_up,Cr_up,C_Ni_up,C_Mn_up,B_Mn_up],axis=1)
    
    final=pd.concat([dk_name,dk_under_to_up,dk_up],axis=1)
    final=pd.DataFrame(final)
    
     
    
    
    
    final=final.replace([np.inf,-np.inf],200000)
    data=final.fillna(0)

    data.to_csv("{}_kernel_on.txt".format(f_name),sep="\t",index=False)
    # # ###### print(data)
    data=pd.read_csv("{}_kernel_on.txt".format(f_name),sep="\t",engine="python",encoding="cp949")
    
    X=data.iloc[:,2:]
    Y=data.iloc[:,0]
    Y2=data.iloc[:,1]
    SS= StandardScaler()
    X_scaled= SS.fit_transform(X)
    pca=PCA(n_components=2)
    pca_result=pca.fit_transform(X_scaled)
    # pca_result=pca.fit_transform(X)
    print(pca.explained_variance_ratio_)
    with open("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/이상하다/{}_New_Standard_scaled2.pkl".format(f_name),'wb') as f :
        pickle.dump(pca,f,pickle.HIGHEST_PROTOCOL)
    pca_pd = pd.DataFrame(pca_result,columns=["PCA1","PCA2"])
    pca_pd['label'] = Y
    pca_pd["class"] = Y2
    pca_pd.to_csv("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/이상하다/{}_New_Standard_scaled2.csv".format(f_name),index=False)
    # pca_pd.to_csv("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/이상하다/{}_Raw_New_Standard_scaled2.txt".format(f_name),sep="\t",index=False)
    cumsum_SS = np.cumsum(pca.explained_variance_ratio_)
    SS_Di = np.argmax(cumsum_SS >=0.95) +1
    print("SS 차원의 수 :" ,SS_Di)
    
    
    
    
    ##### evaluate
                
    base= pickle.load(open("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/이상하다/{}_New_Standard_scaled2.pkl".format(f_name) ,'rb'))
    result = pd.read_csv("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/이상하다/{}_New_Standard_scaled2.csv".format(f_name))
    
    # input_file=pd.read_csv("E:/공간이없두/JASO/input_file2_{}_{}_{}.csv".format(f_name,times_name,file_name))
    siba=input()
    input_file=pd.read_csv("E:/공간이없두/JASO/test_{}.csv".format(siba))
    input_file=input_file.fillna(0) 
    input_file=input_file.iloc[:,:-1]

    input_file['dummy']= 1    
    input_file['dummy3']= 0.1    
        
    C_D=input_file.loc[:,"C"].mul(input_file.loc[:,"C"])
    C_D=pd.DataFrame({"C_D":C_D})
    Mn_under = input_file.loc[:,"Mn"].mul(input_file.loc[:,"dummy"])
    Mn_under = pd.DataFrame({"Mn_under":Mn_under})
    Cr_under = input_file.loc[:,"Cr"].mul(input_file.loc[:,"dummy"])
    Cr_under = pd.DataFrame({"Cr_under":Cr_under})
    C_B = input_file.loc[:,"C"].mul(input_file.loc[:,"B"])
    C_B = pd.DataFrame({"C_B":C_B})
    C_Ni = input_file.loc[:,"C"].mul(input_file.loc[:,"Ni"])
    C_Ni = pd.DataFrame({"C_Ni":C_Ni})
    Mo_Ni = input_file.loc[:,"Mo"].mul(input_file.loc[:,"Ni"])
    Mo_Ni = pd.DataFrame({"Mo_Ni":Mo_Ni})
    V_Ni = input_file.loc[:,"V"].mul(input_file.loc[:,"Ni"])
    V_Ni = pd.DataFrame({"V_Ni":V_Ni})
    
    B_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"B"])
    B_up = pd.DataFrame({"B_up":B_up})
    V_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"V"])
    V_up = pd.DataFrame({"V_up":V_up})
    Cu_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"Cu"])
    Cu_up = pd.DataFrame({"Cu_up":Cu_up})
    Si_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"Si"])
    Si_up = pd.DataFrame({"Si_up":Si_up})
    ###### 01 20 
    C_Mn = input_file.loc[:,"C"].mul(input_file.loc[:,"Mn"])
    C_Mn = pd.DataFrame({"C_Mn":C_Mn})
    B_Mn = input_file.loc[:,"B"].mul(input_file.loc[:,"Mn"])
    B_Mn = pd.DataFrame({"B_Mn":B_Mn})
    
    input_file_sub=pd.concat([input_file,Mn_under],axis=1)
    Mn_C = input_file_sub.loc[:,"Mn_under"].mul(input_file_sub.loc[:,"C"])
    Mn_C = pd.DataFrame({"Mn_C":Mn_C})
       
       
    input_file_name  = input_file.drop(["dummy","dummy3","Pb","N","Nb"],axis=1)
    
    
    input_file_under=pd.concat([C_D,Mn_under,Cr_under,C_B,C_Ni,Mo_Ni,V_Ni,C_Mn,B_Mn,Mn_C],axis=1)    
    input_file_up=pd.concat([B_up,V_up,Cu_up,Si_up],axis=1)
    
    
    input_file_under["dummy"] = 1
    C_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"C_D"])
    C_up = pd.DataFrame({"C_up":C_up})
    Mn_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Mn_under"])
    Mn_up = pd.DataFrame({"Mn_up":Mn_up})
    Cr_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Cr_under"])
    Cr_up = pd.DataFrame({"Cr_up":Cr_up})
    C_B_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"C_B"])
    C_B_up = pd.DataFrame({"C_B_up":C_B_up})
    C_Ni_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"C_Ni"])
    C_Ni_up = pd.DataFrame({"C_Ni_up":C_Ni_up})
    Mo_Ni_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Mo_Ni"])
    Mo_Ni_up = pd.DataFrame({"Mo_Ni_up":Mo_Ni_up})
    V_Ni_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"V_Ni"])
    V_Ni_up = pd.DataFrame({"V_Ni_up":V_Ni_up})
    ### C_Mn ==  걍 곱하기 , Mn_C는 0.1배
    C_Mn_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"C_Mn"])
    C_Mn_up = pd.DataFrame({"C_Mn_up":C_Mn_up})
    B_Mn_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"B_Mn"])
    B_Mn_up = pd.DataFrame({"B_Mn_up":B_Mn_up})
    Mn_C_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Mn_C"])
    Mn_C_up = pd.DataFrame({"Mn_C_up":Mn_C_up})
    
    
    
    
    # input_file_under_to_up = pd.concat([C_up,Mn_up,Cr_up,C_B_up,C_Ni_up,Mo_Ni_up,V_Ni_up],axis=1)
    input_file_under_to_up = pd.concat([C_up,Mn_up,Cr_up,C_Ni_up,C_Mn_up,B_Mn_up],axis=1)
    
    final=pd.concat([input_file_name,input_file_under_to_up,input_file_up],axis=1)
    final=pd.DataFrame(final)
    # print(final)
    final=final.replace([np.inf,-np.inf],200000)
    
    
    # final=input_file.loc[:,:"N"]
    
    
    
    final=SS.transform(final)


    pca_target = base.transform(final)
    
    dt1=[]
    print(type(dt1))
    dt2=[]
    dt3=[]
    dt4=[]
    dt5=[]
    dt6=[]
    dt7=[]
    dt8=[]
    dt9=[]
    dt10=[]
    
    
    for k in range(input_file.shape[0]):       
        target1 = pca_target [k][0]
        target2 = pca_target [k][1]
        print(target1,target2) 
        final=pd.DataFrame()
        
        def euclidean_distance(target1,target2,standard1,standard2) :
            distance=0
            distance1=((target1-standard1)) ** 2
            distance2=((target2-standard2)) ** 2
            distance += (distance1+distance2)      
            return distance **0.5
        lists=pd.DataFrame()
        result=result.groupby(["label"],as_index=False).mean()
        # sns.lmplot("PCA1", "PCA2", data=pca_pd , fit_reg=False, scatter_kws={"s":4}, hue="label")
        # plt.show()


        for i in range(0,result.shape[0]) :
            # print(i+'/'+result.shape[0]-1)
            print(i,'/',result.shape[0])
            name=result.loc[i,"label"]
            standard1=result.loc[i,"PCA1"]
            standard2=result.loc[i,"PCA2"]           
            geori=pd.DataFrame([{"label":name,"distance":euclidean_distance(target1,target2,standard1,standard2)}])
            lists=lists.append(geori)
        final=final.append(lists)

        dk=final.sort_values(by="distance")
        # dk.to_csv("GB_NEW_NOT_scale_with__input_distance.csv")
        dt=dk.iloc[:10,:]
        
        dt1.append(dt.iloc[0,0])
        dt2.append(dt.iloc[1,0])
        dt3.append(dt.iloc[2,0])
        dt4.append(dt.iloc[3,0])
        dt5.append(dt.iloc[4,0])
        dt6.append(dt.iloc[5,0])
        dt7.append(dt.iloc[6,0])
        dt8.append(dt.iloc[7,0])
        dt9.append(dt.iloc[8,0])
        dt10.append(dt.iloc[9,0])
        
        
        # dt=pd.concat([dt,input_file],axis=0)
        # dt.to_csv("E:/공간이없두/JASO/{}_NEW_Standard_scale_with__input_distance_reversion_{}_{}.csv".format(f_name,times_name,file_name),index=False)
    
    input_file.insert(0,"rank1",dt1)
    input_file.insert(0,"rank2",dt2)
    input_file.insert(0,"rank3",dt3)
    input_file.insert(0,"rank4",dt4)
    input_file.insert(0,"rank5",dt5)
    input_file.insert(0,"rank6",dt6)
    input_file.insert(0,"rank7",dt7)
    input_file.insert(0,"rank8",dt8)
    input_file.insert(0,"rank9",dt9)
    input_file.insert(0,"rank10",dt10)
    
    input_file.to_csv("{}_result.csv".format(siba),index=False)
    

if sc == 3 :
    print("Standard gogo")
    print('csv file name = ')
    f_name = "JIS"
    # print("몇번째시도인가요?")
    # times_name= "ASCM320H" 
    times_name = "test_1_evaluation"
    # file_name = input()
    # dk= pd.read_csv("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/{}_Test.txt".format(f_name),sep="\t",engine="python",encoding="cp949")
    data = pd.read_csv("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/{}_Test.txt".format(f_name),sep="\t",engine="python",encoding="cp949")
    ## dummy 제작     
    
    # dk['dummy']= 1    
    # dk['dummy3']= 0.1    
    
    # C_D=dk.loc[:,"C"].mul(dk.loc[:,"C"])
    # C_D=pd.DataFrame({"C_D":C_D})
    # Mn_under = dk.loc[:,"Mn"].mul(dk.loc[:,"dummy"])
    # Mn_under = pd.DataFrame({"Mn_under":Mn_under})
    # Cr_under = dk.loc[:,"Cr"].mul(dk.loc[:,"dummy"])
    # Cr_under = pd.DataFrame({"Cr_under":Cr_under})
    # C_B = dk.loc[:,"C"].mul(dk.loc[:,"B"])
    # C_B = pd.DataFrame({"C_B":C_B})
    # C_Ni = dk.loc[:,"C"].mul(dk.loc[:,"Ni"])
    # C_Ni = pd.DataFrame({"C_Ni":C_Ni})
    # Mo_Ni = dk.loc[:,"Mo"].mul(dk.loc[:,"Ni"])
    # Mo_Ni = pd.DataFrame({"Mo_Ni":Mo_Ni})
    # V_Ni = dk.loc[:,"V"].mul(dk.loc[:,"Ni"])
    # V_Ni = pd.DataFrame({"V_Ni":V_Ni})
    
    # B_up = dk.loc[:,"dummy"].div(dk.loc[:,"B"])
    # B_up = pd.DataFrame({"B_up":B_up})
    # V_up = dk.loc[:,"dummy"].div(dk.loc[:,"V"])
    # V_up = pd.DataFrame({"V_up":V_up})
    # Cu_up = dk.loc[:,"dummy"].div(dk.loc[:,"Cu"])
    # Cu_up = pd.DataFrame({"Cu_up":Cu_up})
    # Si_up = dk.loc[:,"dummy"].div(dk.loc[:,"Si"])
    # Si_up = pd.DataFrame({"Si_up":Si_up})
       
    
    # dk_name = dk.drop(["dummy","dummy3","Al","Ti","Nb","W","Pb","N"],axis=1)
    
    
    # dk_under=pd.concat([C_D,Mn_under,Cr_under,C_B,C_Ni,Mo_Ni,V_Ni],axis=1)    
    # dk_up=pd.concat([B_up,V_up,Cu_up,Si_up],axis=1)
    
    
    # dk_under["dummy"] = 1
    # C_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"C_D"])
    # C_up = pd.DataFrame({"C_up":C_up})
    # Mn_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Mn_under"])
    # Mn_up = pd.DataFrame({"Mn_up":Mn_up})
    # Cr_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Cr_under"])
    # Cr_up = pd.DataFrame({"Cr_up":Cr_up})
    # C_B_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"C_B"])
    # C_B_up = pd.DataFrame({"C_B_up":C_B_up})
    # C_Ni_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"C_Ni"])
    # C_Ni_up = pd.DataFrame({"C_Ni_up":C_Ni_up})
    # Mo_Ni_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"Mo_Ni"])
    # Mo_Ni_up = pd.DataFrame({"Mo_Ni_up":Mo_Ni_up})
    # V_Ni_up = dk_under.loc[:,"dummy"].div(dk_under.loc[:,"V_Ni"])
    # V_Ni_up = pd.DataFrame({"V_Ni_up":V_Ni_up})
    
    
    
    
    # # dk_under_to_up = pd.concat([C_up,Mn_up,Cr_up,C_B_up,C_Ni_up,Mo_Ni_up,V_Ni_up],axis=1)
    # dk_under_to_up = pd.concat([C_up,Mn_up,Cr_up],axis=1)
    
    # final=pd.concat([dk_name,dk_under_to_up,dk_up],axis=1)
    # final=pd.DataFrame(final)
    
     
    
    
    
    # final=final.replace([np.inf,-np.inf],200000)
    # data=final.fillna(0)

    # data.to_csv("{}_kernel_on.txt".format(f_name),sep="\t",index=False)
    # # # ###### print(data)
    # data=pd.read_csv("{}_kernel_on.txt".format(f_name),sep="\t",engine="python",encoding="cp949")
    
    X=data.iloc[:,2:]
    Y=data.iloc[:,0]
    Y2=data.iloc[:,1]
    SS= StandardScaler()
    X_scaled= SS.fit_transform(X)
    pca=PCA(n_components=2)
    pca_result=pca.fit_transform(X_scaled)
    # pca_result=pca.fit_transform(X)
    print(pca.explained_variance_ratio_)
    with open("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/이상하다/{}_New_Standard_scaled.pkl".format(f_name),'wb') as f :
        pickle.dump(pca,f,pickle.HIGHEST_PROTOCOL)
    pca_pd = pd.DataFrame(pca_result,columns=["PCA1","PCA2"])
    pca_pd['label'] = Y
    pca_pd["class"] = Y2
    pca_pd.to_csv("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/이상하다/{}_New_Standard_scaled.csv".format(f_name),index=False)
    # pca_pd.to_csv("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/이상하다/{}_Raw_New_Standard_scaled.txt".format(f_name),sep="\t",index=False)
    cumsum_SS = np.cumsum(pca.explained_variance_ratio_)
    SS_Di = np.argmax(cumsum_SS >=0.95) +1
    print("SS 차원의 수 :" ,SS_Di)
    
    
    
    
    ##### evaluate
                
    base= pickle.load(open("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/이상하다/{}_New_Standard_scaled.pkl".format(f_name) ,'rb'))
    result = pd.read_csv("C:/Users/관리자/Desktop/python worksapce/After12_23/New Db/이상하다/{}_New_Standard_scaled.csv".format(f_name))
    
    # input_file=pd.read_csv("E:/공간이없두/JASO/input_file2_{}_{}_{}.csv".format(f_name,times_name,file_name))
    siba=input()
    input_file=pd.read_csv("E:/공간이없두/JASO/test_{}.csv".format(siba))
    input_file=input_file.fillna(0) 
    input_file=input_file.iloc[:,:-1]

    # input_file['dummy']= 1    
    # input_file['dummy3']= 0.1    
        
    # C_D=input_file.loc[:,"C"].mul(input_file.loc[:,"C"])
    # C_D=pd.DataFrame({"C_D":C_D})
    # Mn_under = input_file.loc[:,"Mn"].mul(input_file.loc[:,"dummy"])
    # Mn_under = pd.DataFrame({"Mn_under":Mn_under})
    # Cr_under = input_file.loc[:,"Cr"].mul(input_file.loc[:,"dummy"])
    # Cr_under = pd.DataFrame({"Cr_under":Cr_under})
    # C_B = input_file.loc[:,"C"].mul(input_file.loc[:,"B"])
    # C_B = pd.DataFrame({"C_B":C_B})
    # C_Ni = input_file.loc[:,"C"].mul(input_file.loc[:,"Ni"])
    # C_Ni = pd.DataFrame({"C_Ni":C_Ni})
    # Mo_Ni = input_file.loc[:,"Mo"].mul(input_file.loc[:,"Ni"])
    # Mo_Ni = pd.DataFrame({"Mo_Ni":Mo_Ni})
    # V_Ni = input_file.loc[:,"V"].mul(input_file.loc[:,"Ni"])
    # V_Ni = pd.DataFrame({"V_Ni":V_Ni})
    
    # B_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"B"])
    # B_up = pd.DataFrame({"B_up":B_up})
    # V_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"V"])
    # V_up = pd.DataFrame({"V_up":V_up})
    # Cu_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"Cu"])
    # Cu_up = pd.DataFrame({"Cu_up":Cu_up})
    # Si_up = input_file.loc[:,"dummy"].div(input_file.loc[:,"Si"])
    # Si_up = pd.DataFrame({"Si_up":Si_up})
       
    
    # input_file_name = input_file.drop(["dummy","dummy3","Al","Ti","Nb","W","Pb","N"],axis=1)
    
    
    # input_file_under=pd.concat([C_D,Mn_under,Cr_under,C_B,C_Ni,Mo_Ni,V_Ni],axis=1)    
    # input_file_up=pd.concat([B_up,V_up,Cu_up,Si_up],axis=1)
    
    
    # input_file_under["dummy"] = 1
    # C_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"C_D"])
    # C_up = pd.DataFrame({"C_up":C_up})
    # Mn_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Mn_under"])
    # Mn_up = pd.DataFrame({"Mn_up":Mn_up})
    # Cr_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Cr_under"])
    # Cr_up = pd.DataFrame({"Cr_up":Cr_up})
    # C_B_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"C_B"])
    # C_B_up = pd.DataFrame({"C_B_up":C_B_up})
    # C_Ni_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"C_Ni"])
    # C_Ni_up = pd.DataFrame({"C_Ni_up":C_Ni_up})
    # Mo_Ni_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"Mo_Ni"])
    # Mo_Ni_up = pd.DataFrame({"Mo_Ni_up":Mo_Ni_up})
    # V_Ni_up = input_file_under.loc[:,"dummy"].div(input_file_under.loc[:,"V_Ni"])
    # V_Ni_up = pd.DataFrame({"V_Ni_up":V_Ni_up})
    
    
    
    
    # # input_file_under_to_up = pd.concat([C_up,Mn_up,Cr_up,C_B_up,C_Ni_up,Mo_Ni_up,V_Ni_up],axis=1)
    # input_file_under_to_up = pd.concat([C_up,Mn_up,Cr_up],axis=1)
    
    # final=pd.concat([input_file_name,input_file_under_to_up,input_file_up],axis=1)
    # final=pd.DataFrame(final)
    # # print(final)
    # final=final.replace([np.inf,-np.inf],200000)
    
    
    final=input_file.loc[:,:"N"]
    
    
    
    # final=SS.transform(final)


    pca_target = base.transform(final)
    
    dt1=[]
    print(type(dt1))
    dt2=[]
    dt3=[]
    dt4=[]
    dt5=[]
    dt6=[]
    dt7=[]
    dt8=[]
    dt9=[]
    dt10=[]
    
    
    for k in range(input_file.shape[0]):       
        target1 = pca_target [k][0]
        target2 = pca_target [k][1]
        print(target1,target2) 
        final=pd.DataFrame()
        
        def euclidean_distance(target1,target2,standard1,standard2) :
            distance=0
            distance1=((target1-standard1)) ** 2
            distance2=((target2-standard2)) ** 2
            distance += (distance1+distance2)      
            return distance **0.5
        lists=pd.DataFrame()
        result=result.groupby(["label"],as_index=False).mean()
        # sns.lmplot("PCA1", "PCA2", data=pca_pd , fit_reg=False, scatter_kws={"s":4}, hue="label")
        # plt.show()


        for i in range(0,result.shape[0]) :
            # print(i+'/'+result.shape[0]-1)
            print(i,'/',result.shape[0])
            name=result.loc[i,"label"]
            standard1=result.loc[i,"PCA1"]
            standard2=result.loc[i,"PCA2"]           
            geori=pd.DataFrame([{"label":name,"distance":euclidean_distance(target1,target2,standard1,standard2)}])
            lists=lists.append(geori)
        final=final.append(lists)

        dk=final.sort_values(by="distance")
        # dk.to_csv("GB_NEW_NOT_scale_with__input_distance.csv")
        dt=dk.iloc[:10,:]
        
        dt1.append(dt.iloc[0,0])
        dt2.append(dt.iloc[1,0])
        dt3.append(dt.iloc[2,0])
        dt4.append(dt.iloc[3,0])
        dt5.append(dt.iloc[4,0])
        dt6.append(dt.iloc[5,0])
        dt7.append(dt.iloc[6,0])
        dt8.append(dt.iloc[7,0])
        dt9.append(dt.iloc[8,0])
        dt10.append(dt.iloc[9,0])
        
        
        # dt=pd.concat([dt,input_file],axis=0)
        # dt.to_csv("E:/공간이없두/JASO/{}_NEW_Standard_scale_with__input_distance_reversion_{}_{}.csv".format(f_name,times_name,file_name),index=False)
    
    input_file.insert(0,"rank1",dt1)
    input_file.insert(0,"rank2",dt2)
    input_file.insert(0,"rank3",dt3)
    input_file.insert(0,"rank4",dt4)
    input_file.insert(0,"rank5",dt5)
    input_file.insert(0,"rank6",dt6)
    input_file.insert(0,"rank7",dt7)
    input_file.insert(0,"rank8",dt8)
    input_file.insert(0,"rank9",dt9)
    input_file.insert(0,"rank10",dt10)
    
    input_file.to_csv("{}_result.csv".format(siba),index=False)