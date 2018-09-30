import matplotlib.pyplot as plt

fig = plt.figure(figsize=(50,40))
#plt.suptitle('Experimenting with the batch_size parameter on the CNN',fontsize=50)
plt.suptitle('Experimenting with the dropout_rate parameter on the CNN',fontsize=50)
ax = fig.add_subplot(111)
ax.set_xlabel('dropout rate',fontsize=40)
ax.set_ylabel('PSNR',fontsize=40)

#change ticks size
ax.tick_params(axis='both',labelsize=20)

#move x and y labels downwards
ax.xaxis.labelpad = 50
ax.yaxis.labelpad = 50


# line1=plt.plot([20,40,60,80,100,120], [21.9463796468,22.1433124857,22.1671861216,22.2402920249,22.296930919,22.3137967568],'bo-',linewidth=3,label='image inpainting')
# line2=plt.plot([20,40,60,80,100,120], [19.8118409303,20.2802053634,20.4400316665,20.8078954383,20.7730720941,20.7435289814],'rv-',linewidth=3,label='image denoising')
line1=plt.plot([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95], [22.4213868219,22.3783729208,22.296930919,22.0737570023,21.9076688838,21.6867925406,21.2336318228,20.0596825281,19.1264353957],'bo-',linewidth=3,label='image inpainting')
line2=plt.plot([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95], [20.9485768944,20.8883746542,20.7730720941,20.5984768509,20.2987485786,20.0127384752,19.7675532455,18.7986765784,17.989768798],'rv-',linewidth=3,label='image denoising')


#add labels to lines
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.,prop={'size':30})


plt.show()