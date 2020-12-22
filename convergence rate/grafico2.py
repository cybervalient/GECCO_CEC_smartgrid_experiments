import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from numpy import array

fig, ax = plt.subplots(1, 1, figsize=(12,6)) # make the figure with the size 10 x 6 inches
fig.suptitle('Average Convergence Rate')

x = np.linspace(0,20,20)

VNS_DEEPSO = [1.931772257,
1.928934636,
1.912381266,
1.91229119,
1.928390159,
1.932595323,
1.926461694,
1.93068558,
1.911056131,
1.925231314,
1.894835989,
1.908185714,
1.930659602,
1.887184495,
1.931138363,
1.913192487,
1.927253087,
1.927974932,
1.909941412,
1.914210011
]


HL_PS_VNSO = [-2.309799836,
-2.482282395,
-2.184189811,
-2.513037568,
-2.421680954,
-2.551191458,
-2.073749097,
-2.473257646,
-2.615835457,
-1.444377104,
-1.846580895,
-2.227832362,
-2.123289337,
-2.59155454,
-2.059388135,
-2.254040179,
-1.899730351,
-2.60245018,
-2.057676135,
-2.651723075
]


GM_VNPSO = [-5.272562475,
-4.451855456,
-4.507334083,
-4.373935865,
-4.278370921,
-4.209388405,
-4.800633094,
-4.089025451,
-3.72485308,
-4.779849819,
-4.846137164,
-4.99326779,
-4.927845352,
-4.294308397,
-4.784427671,
-4.727140035,
-4.731351495,
-3.820123024,
-5.007529989,
-4.443695465
]


PSO_GBP = [-0.068825896,
-0.277960803,
-0.174545293,
-0.273856113,
-0.272077497,
-0.282353909,
-0.167935315,
-0.170124258,
-0.276271494,
-0.269794898,
-0.286424549,
-0.173492296,
-0.267937446,
-0.276906775,
-0.165995976,
-0.173198325,
-0.274403393,
-0.271229697,
-0.274733032,
-0.268276779
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
line_labels = ["CUMDANCauchy", "PSO-GBP", "GM-VNPSO","HL-PSVNSO","VNS-DEEPSO"]


l1 = ax.plot(x,CUMDANCAUCHY)[0]
l2 = ax.plot(x,PSO_GBP)[0]
l3 = ax.plot(x,GM_VNPSO)[0]
l4 = ax.plot(x,HL_PS_VNSO)[0]
l5 = ax.plot(x,VNS_DEEPSO)[0]




lgd = fig.legend([l1, l2, l3, l4, l5],              # List of the line objects
           labels= line_labels,       # The labels for each line
           loc="center right",        # Position of the legend
           borderaxespad=0.1,         # Add little spacing around the legend box
         #  bbox_to_anchor=(1.3, 0.5),
           title="2019 Algorithm"        # Title for the legend
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