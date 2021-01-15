import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from numpy import array

fig, ax = plt.subplots(1, 1, figsize=(12,6)) # make the figure with the size 10 x 6 inches
fig.suptitle('Average Convergence Rate')

x = np.linspace(0,20,20)

ABC_DE = [-2.020192858,
-1.070524907,
-1.041318444,
-1.023591748,
-0.065826594,
-1.102230607,
-0.023634273,
-0.990196634,
-1.083720729,
-0.051494113,
-0.058652866,
-0.031233368,
-0.01979443,
-0.023090505,
-1.027811494,
-1.058626264,
-1.076333853,
-0.019140621,
-0.019358197,
-1.050592831
]


HFEABC = [-53.57882332,
-34.46454482,
-62.42290464,
-52.19644691,
-53.52428241,
-51.71191708,
-51.7137871,
-34.68260509,
-52.47143788,
-62.90427259,
-49.6978036,
-50.46992391,
-51.65121768,
-53.60441976,
-60.14823096,
-42.60447409,
-51.57006898,
-51.64905941,
-51.93167139,
-35.69291095
]


CE_CMAES = [-0.444345066,
-0.438943645,
-0.46229298,
-0.035021705,
-0.662016398,
-0.637474925,
-0.245005673,
-0.440733022,
-0.243341209,
-0.247817239,
-0.23094018,
0.149100448,
-0.233314321,
-0.437657996,
-0.851960792,
-0.848396067,
-0.458222386,
-0.051833616,
-0.242199542,
-0.238085679
]


AJSO = [0.127783301,
0.134378224,
0.128604572,
0.128698818,
0.326471944,
0.318539354,
0.128571559,
0.13026274,
0.129752469,
0.122347401,
0.124778272,
0.138600852,
0.139037302,
0.127633056,
0.304979824,
0.128125637,
0.116929456,
0.136015436,
0.121972264,
0.322477733
]


GASAPSO = [-0.731213287,
-0.720475601,
-0.74295464,
-0.959732666,
-0.938621669,
-0.922494287,
-1.141265654,
-0.740666058,
-0.514534799,
-1.148349028,
-0.75992326,
-0.714485355,
-0.732464525,
-0.727824185,
-0.758591988,
-0.720856167,
-0.736771934,
-0.943341662,
-0.53293284,
-0.514323466
]



CUMDANCAUCHY = [-0.176021556,
-0.260756485,
-0.371070449,
-0.608703668,
-0.28948412,
-0.178547893,
-0.398915601,
-0.221171608,
-0.261796283,
-0.331912487,
-0.117848412,
-0.317197835,
-0.483445847,
-0.18919594,
-0.069522334,
-0.281743617,
-0.300508147,
-0.322299167,
-0.418225806,
-0.341571473
]


# Labels to use for each line
line_labels = ["CUMDANCauchy", "ABC-DE", "HFEABC","CE-CMAES","GASAPSO","AJSO"]


l1 = ax.plot(x,CUMDANCAUCHY)[0]
l2 = ax.plot(x,ABC_DE)[0]
l3 = ax.plot(x,HFEABC)[0]
l4 = ax.plot(x,CE_CMAES)[0]
l5 = ax.plot(x,AJSO)[0]
l6 = ax.plot(x,GASAPSO)[0]



lgd = fig.legend([l1, l2, l3, l4, l5, l6],              # List of the line objects
           labels= line_labels,       # The labels for each line
           loc="center right",        # Position of the legend
           borderaxespad=0.1,         # Add little spacing around the legend box
         #  bbox_to_anchor=(1.3, 0.5),
           title="2020 Algorithm"        # Title for the legend
           )      



#plt.title('Average Convergence Rate')
plt.xlabel('Run')
plt.ylabel('Rate')
#plt.legend(loc=0)
plt.grid(True)
# Adjust the scaling factor to fit your legend text completely outside the plot
# (smaller value results in more space being made for the legend)
plt.subplots_adjust(right=0.85)
plt.show()

#fig.savefig('image_output.png',
 #           dpi=300, 
  #          format='png', 
   #         bbox_extra_artists=(lgd,),
    #        bbox_inches='tight')