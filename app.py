from tkinter import *
import numpy as np
import pickle
win=Tk()
win.title('Bank Notes Prediction')
win.configure(bg='cyan')
win.geometry('800x400')
def predict():
    var=v1.get()
    skew=s1.get()
    cur=c1.get()
    ent=en1.get()
    a=np.array([[float(var),float(skew),float(cur),float(ent)]])
    with open('bankscaler','rb') as f:
        sc=pickle.load(f)
    with open('bankmodel','rb') as f:
        model=pickle.load(f)
    a_sc=sc.transform(a)
    result=model.predict(a_sc)
    if result[0]==1:
        x='ITS A FAKE NOTE'
    else:
        x='ITS A REAL NOTE'
    l5.config(text=x)
    v1.set('')
    s1.set('')
    c1.set('')
    en1.set('')


l0=Label(win,text='BANK NOTES PREDICTION SYSTEM',bd=5,
         relief='ridge',bg='white',fg='black',width=45,font=('times new roman',18,'bold'))

l0.place(x=280,y=40)

l1=Label(win,text='VARIANCE',bd=5,
         relief='ridge',bg='white',fg='black',width=25,font=('times new roman',12,'bold'))

l1.place(x=100,y=140)

v1=StringVar()
e1=Entry(win,textvariable=v1,bd=5,
         relief='ridge',bg='white',fg='black',width=40,font=('times new roman',12,'bold'))

e1.place(x=350,y=140)
# skewness
l2=Label(win,text='SKEWNESS',bd=5,
         relief='ridge',bg='white',fg='black',width=25,font=('times new roman',12,'bold'))

l2.place(x=100,y=180)

s1=StringVar()
e2=Entry(win,textvariable=s1,bd=5,
         relief='ridge',bg='white',fg='black',width=40,font=('times new roman',12,'bold'))

e2.place(x=350,y=180)
# curtosis

l3=Label(win,text='CURTOSIS',bd=5,
         relief='ridge',bg='white',fg='black',width=25,font=('times new roman',12,'bold'))

l3.place(x=100,y=220)

c1=StringVar()
e3=Entry(win,textvariable=c1,bd=5,
         relief='ridge',bg='white',fg='black',width=40,font=('times new roman',12,'bold'))

e3.place(x=350,y=220)
#Entropy
l4=Label(win,text='ENTROPY',bd=5,
         relief='ridge',bg='white',fg='black',width=25,font=('times new roman',12,'bold'))

l4.place(x=100,y=260)

en1=StringVar()
e4=Entry(win,textvariable=en1,bd=5,
         relief='ridge',bg='white',fg='black',width=40,font=('times new roman',12,'bold'))

e4.place(x=350,y=260)

# predict
b1=Button(win,text='PREDICT',width=20,command=predict)
b1.place(x=300,y=320)

# label
l5=Label(win,text='Fake or Real ??',bd=5,
         relief='ridge',bg='white',
         fg='black',width=20,height=5,font=('times new roman',12,'bold'))

l5.place(x=600,y=320)

win.mainloop()