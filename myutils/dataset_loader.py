import os
import pandas as pd



DATASETS={


"NASA":[
"CM1",
"JM1",
"KC1",
"KC2",
"MW1",
"PC1",
"PC3",
"PC4"
],


"AEEEM":[
"EQ",
"JDT",
"LC",
"ML",
"PDE"
],


"ReLink":[
"Apache",
"Safe",
"ZXing"
]

}



def load_dataset():

    datasets=[]


    for group,names in DATASETS.items():


        for name in names:


            path=os.path.join(
                "datasets",
                group,
                name+".csv"
            )


            datasets.append(
                (
                group,
                name,
                path
                )
            )


    return datasets