"""Reproduce journal figures from retained measured evidence. Requires numpy, matplotlib."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path(__file__).resolve().parents[1]
O=R/'figures'
M=R/'training/results/2026-10-28/metrics.json'
m=json.loads(M.read_text()); labels=m['labels']; names=['Graphics card','Motherboard','RAM','SSD']
p=json.loads((O/'evidence/test_predictions.json').read_text())
y=np.array([labels.index(x['true']) for x in p]); prob=np.array([x['probabilities'] for x in p]); pred=prob.argmax(1)
cm=np.zeros((4,4),int)
np.add.at(cm,(y,pred),1)
assert len(p)==145 and np.array_equal(cm,m['fine_tuned_test']['confusion_matrix'])
assert abs((pred==y).mean()-m['fine_tuned_test']['accuracy'])<1e-12
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
manifest=[]
def save(fig,name,caption):
 fig.tight_layout(rect=(0,.065,1,.94));fig.text(.5,.018,'Hardware recognition only • exploratory image split • physical-item IDs unavailable',ha='center',fontsize=8)
 for ext in ['png','pdf','svg']:fig.savefig(O/(name+'.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
 manifest.append({'file':name,'caption':caption});plt.close(fig)
# Actual architecture, not an experimental metric.
f,ax=plt.subplots(figsize=(10,5));ax.axis('off')
boxes=[(.02,.66,'Browser\nOriginal photos / review'),(.36,.66,'Node 1 · Gateway\nRules, jobs and signed history'),(.1,.26,'Node 2 · Policy\nLexical retrieval + local text LLM'),(.63,.26,'Node 3 · Vision\nLocal vision LLM + ONNX hint'),(.37,.02,'Local Ollama\nText / vision model weights')]
for x,z,t in boxes:
 ax.add_patch(FancyBboxPatch((x,z),.31,.18,boxstyle='round,pad=.012',fc='#eaf1f8',ec='#31577b'));ax.text(x+.155,z+.09,t,ha='center',va='center',fontsize=10)
for a,b in [((.33,.75),(.36,.75)),((.46,.65),(.26,.45)),((.58,.65),(.77,.45)),((.27,.25),(.42,.2)),((.76,.25),(.63,.2))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#31577b','lw':1.7})
f.suptitle('SmartRMA: implemented three-node workflow');save(f,'fig0_system_architecture','Implemented local workflow. Gateway applies deterministic rules; the classifier provides product-type advice only. T4 values above USD 2,500 require a human decision. Arrows are architecture, not measured telemetry.')
f,axs=plt.subplots(1,2,figsize=(10,4.6))
for ax,key,title in zip(axs,['frozen_imagenet_centroid_baseline_test','fine_tuned_test'],['Frozen ImageNet centroid baseline','Fine-tuned MobileNetV3 Small']):
 c=np.array(m[key]['confusion_matrix']);ax.imshow(c,cmap='Blues',vmin=0,vmax=56)
 for (i,j),v in np.ndenumerate(c):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>28 else 'black')
 ax.set(xticks=range(4),yticks=range(4),xticklabels=['GPU','Board','RAM','SSD'],yticklabels=names,xlabel='Predicted type',ylabel='True source label',title=title)
f.suptitle('Hardware-type confusion matrices (same 145 test images)');save(f,'fig1_confusion_matrix','Counts from the retained 145-image test split; rows are source labels and columns predicted hardware types. Neither matrix evaluates damage or return decisions.')
f,ax=plt.subplots(figsize=(8,4.5));x=np.arange(4)
for i,(key,title) in enumerate([('precision','Precision'),('recall','Recall'),('f1-score','F1')]):
 vals=[m['fine_tuned_test']['classification_report'][l][key]*100 for l in labels];bars=ax.bar(x+(i-1)*.24,vals,.24,label=title)
 ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
ax.set(xticks=x,xticklabels=[f'{n}\n(n={sum(y==i)})' for i,n in enumerate(names)],ylim=(0,114),ylabel='Percent');ax.legend(ncol=3);f.suptitle('Fine-tuned hardware recognition: per-class metrics');save(f,'fig2_precision_recall_f1','Per-class precision, recall and F1 on 145 images. GPU recall is 52.17%; SSD 19/19 is a small, source-limited sample and does not establish general perfect performance.')
# Exact empirical one-vs-rest ROC/PR using stored probabilities; tied scores grouped.
f,axs=plt.subplots(1,2,figsize=(10,4.5));curve={}
for k,name in enumerate(names):
 truth=(y==k).astype(int);score=prob[:,k];order=np.argsort(-score,kind='stable');t=truth[order];s=score[order];end=np.r_[np.where(np.diff(s)!=0)[0],len(s)-1];tp=np.cumsum(t)[end];fp=(end+1)-tp
 recall=np.r_[0,tp/truth.sum()];fpr=np.r_[0,fp/(len(t)-truth.sum())];precision=tp/(tp+fp);auc=float(np.trapezoid(recall,fpr));ap=float(np.sum(np.diff(recall)*precision));curve[labels[k]]={'roc_auc_ovr':auc,'average_precision':ap,'positives':int(truth.sum())}
 axs[0].plot(fpr,recall,label=f'{name}: {auc:.3f}');axs[1].step(recall,np.r_[1,precision],where='pre',label=f'{name}: {ap:.3f}')
axs[0].plot([0,1],[0,1],'k--',lw=.7);axs[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='One-vs-rest ROC (legend: AUC)');axs[1].set(xlabel='Recall',ylabel='Precision',title='One-vs-rest PR (legend: average precision)')
for ax in axs:ax.set(xlim=(0,1),ylim=(0,1.03));ax.legend(fontsize=8,loc='lower left')
f.suptitle('Hardware-type ranking curves · stored test probabilities');save(f,'fig3_roc_pr_curves','Empirical one-vs-rest hardware-type ROC and precision-recall curves computed from stored probabilities for the same 145 test images. These are not damage-detection curves. Average precision uses recall-increment weighting; no confidence intervals or independent replication are claimed.')
f,axs=plt.subplots(1,2,figsize=(9,4));ep=[v['epoch'] for v in m['history']]
axs[0].plot(ep,[v['train_loss'] for v in m['history']],marker='o');axs[0].set(xlabel='Epoch',ylabel='Weighted cross entropy',title='Training loss')
axs[1].plot(ep,[v['val_macro_f1'] for v in m['history']],marker='o');axs[1].axvline(9,color='gray',ls='--',label='Selected epoch 9');axs[1].set(xlabel='Epoch',ylabel='Macro F1',title='Validation hardware recognition');axs[1].legend()
f.suptitle('Recorded Colab training history · 679 train / 147 validation');save(f,'fig4_training_history','Twelve recorded training epochs; checkpoint selected by validation macro F1 at epoch 9. Curves are training loss and validation F1, not training/test accuracy.')
f,ax=plt.subplots(figsize=(8,4.5));x=np.arange(4);bottom=np.zeros(4)
for split,color in [('train','#355c86'),('val','#59a5a0'),('test','#e5ad58')]:
 values=np.array([m['dataset_counts'][split][l] for l in labels]);bars=ax.bar(x,values,bottom=bottom,label=split,color=color);ax.bar_label(bars,label_type='center',fontsize=9);bottom+=values
ax.set(xticks=x,xticklabels=names,ylabel='Original retained images');ax.legend();f.suptitle('Dataset composition: 971 images, four product types');save(f,'fig5_dataset_composition','971 original retained images after duplicate auditing: 679 train, 147 validation and 145 test. Image groups use exact-pixel deduplication and pHash <= 6; physical-item independence is not established.')
f,axs=plt.subplots(1,2,figsize=(9,4));keys=['frozen_imagenet_centroid_baseline_test','fine_tuned_test']
for ax,metric,title in zip(axs,['accuracy','macro_f1'],['Test recognition accuracy','Test macro F1']):
 values=[m[k][metric]*100 for k in keys];bars=ax.bar(['Frozen centroid','Fine-tuned'],values,color=['#8a9daf','#355c86']);ax.bar_label(bars,fmt='%.2f',padding=3);ax.set(ylim=(0,105),ylabel='Percent',title=title)
f.suptitle('Same-split comparison · 145 held-out images');save(f,'fig6_baseline_comparison','Fine-tuned recognition accuracy 122/145 (84.14%) versus frozen-feature centroid baseline 109/145 (75.17%); macro F1 0.8301 versus 0.7339. This baseline is not Qwen3-VL or the old RMA system; GPU recall declined after training.')
f,ax=plt.subplots(figsize=(8,4.5));base=m[keys[0]]['classification_report'];fine=m[keys[1]]['classification_report'];x=np.arange(4)
for offset,report,title,color in [(-.18,base,'Frozen centroid','#8a9daf'),(.18,fine,'Fine-tuned','#355c86')]:
 bars=ax.bar(x+offset,[report[l]['recall']*100 for l in labels],.36,label=title,color=color);ax.bar_label(bars,fmt='%.1f',padding=2,fontsize=8)
ax.set(xticks=x,xticklabels=names,ylim=(0,114),ylabel='Recall (%)');ax.legend();f.suptitle('Per-class recall: improvement is not uniform');save(f,'fig7_recall_tradeoffs','Recall changes on identical test images. GPU recall drops from 82.61% to 52.17%, while RAM recall rises from 51.06% to 85.11%. Report the tradeoff alongside aggregate accuracy.')
(O/'derived_curve_metrics.json').write_text(json.dumps(curve,indent=2)+'\n')
(O/'figure_manifest.json').write_text(json.dumps({'metrics_sha256':hashlib.sha256(M.read_bytes()).hexdigest(),'predictions_sha256':hashlib.sha256((O/'evidence/test_predictions.json').read_bytes()).hexdigest(),'test_count':145,'correct':int(sum(y==pred)),'figures':manifest},indent=2)+'\n')
(O/'CAPTIONS.md').write_text('# Journal figure captions\n\n'+'\n\n'.join(f"**{v['file']}** — {v['caption']}" for v in manifest)+'\n')
print('Generated',len(manifest),'figures in PNG, PDF and SVG; evidence checks passed.')
