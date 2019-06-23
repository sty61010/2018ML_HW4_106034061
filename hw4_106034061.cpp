//
//  hw4_106034061.cpp
//  Program
//
//  Created by 曾靖渝 on 2018/10/27.
//  Copyright © 2018年 曾靖渝. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <math.h>
#define ETA 0.2
#define MAX 100
using namespace std;
using example=vector<double>;
class Neural_Network
{
private:
    int attribute_number;
    int instance_number;
    int class_label_number;
    vector<string> class_labels;
    set<example> training_set;
    double target_vector[MAX];
    double hidden_layer_neurons1[MAX];//Attribute Number
    double hidden_layer_neurons2[MAX];//Attribute Number
    double output_layer_neuron[MAX];//Class_Label Number
    double weight1[MAX][MAX];//Hidden Layer Neuron2 to Output Neuron
    double weight2[MAX][MAX];//Hidden Layer Neuron1 to Hidden Layer Neuron2
    double weight3[MAX][MAX];//Input Signals(Attribute) to Hidden Layer Neuron1
    double responsibility1[MAX];//Responsibility of Output Neuron
    double responsibility2[MAX];//Responsibility of Hidden Layer Neuron2
    double responsibility3[MAX];//Responsibility of Hidden Layer Neuron1
    double MSE_past;
    double MSE_now;
public:
    Neural_Network(int attribute_number ,int instance_number, int class_label_number)
    {
        this->attribute_number=attribute_number;
        this->instance_number=instance_number;
        this->class_label_number=class_label_number;
        MSE_now=0;
        MSE_past=0;
        initial_classification();
        print_weights();
    }
    void initial_classification(void)
    {
        for(int i=0;i<attribute_number;i++)
        {
            for(int j=0;j<class_label_number;j++)
            {
                weight1[i][j]=0.001;
            }
        }
        for(int i=0;i<attribute_number;i++)
        {
            for(int j=0;j<attribute_number;j++)
            {
                weight2[i][j]=0.001;
                weight3[i][j]=0.001;
            }
        }
    }
    void add_to_training_set(example ex, string class_label)
    {
        bool found=false;
        for(int i=0;i<class_labels.size();i++)
        {
            if(class_labels[i]==class_label)
            {
                found=true;
                ex.push_back(i);
                break;
            }
        }
        if(!found)
        {
            ex.push_back(class_labels.size());
            class_labels.push_back(class_label);
        }
        training_set.insert(ex);
        Classification(ex, class_label);
    }
    void Classification(example ex, string class_label)
    {
        print_example(ex);
        examine_target_vector(ex);
//        print_target_vector(ex);
        Forward_Propagation(ex);
//        print_layer_neurons();
//        print_output_signals();
        BackPropagation_Error(ex);
        weight_modification(ex);
//        print_weights();
        MSE_calculation();
        print_MSE();
    }
    void examine_target_vector(example ex)
    {
        for(int i=0;i<class_label_number;i++)
        {
            if(ex[attribute_number]==i)target_vector[i]=1;
            else target_vector[i]=0;
        }
    }
    void Forward_Propagation(example ex)
    {
        //Output of Hidden Layer Neurons2
        for(int i=0;i<attribute_number;i++)
        {
            double z=0;
            for(int j=0;j<attribute_number;j++)
            {
                z+=weight3[j][i]*ex[j];
            }
            z=sigmoid_function(z);
            hidden_layer_neurons2[i]=z;
        }
        //Output of Hidden Layer Neurons1
        for(int i=0;i<attribute_number;i++)
        {
            double z=0;
            for(int j=0;j<attribute_number;j++)
            {
                z+=weight2[j][i]*hidden_layer_neurons2[j];
            }
            z=sigmoid_function(z);
            hidden_layer_neurons1[i]=z;
        }
        //Output of Output Layer Neurons
        for(int i=0;i<class_label_number;i++)
        {
            double z=0;
            for(int j=0;j<attribute_number;j++)
            {
                z+=weight1[j][i]*hidden_layer_neurons1[j];
            }
            z=sigmoid_function(z);
            output_layer_neuron[i]=z;
        }
    }
    double sigmoid_function(double before_calculation)
    {
        double after_calculation=0;
        after_calculation=(double)1/((double)1+(exp(-before_calculation)));
        return after_calculation;
    }
    void BackPropagation_Error(example ex)
    {
        //Output of Output Layer Neurons(Responsibility1)
        for(int i=0;i<class_label_number;i++)
        {
            double y=output_layer_neuron[i];
            double res=y*(1-y)*(target_vector[i]-y);
            responsibility1[i]=res;
        }
        //Output of Hidden Layer Neurons1(Responsibility2)
        for(int i=0;i<attribute_number;i++)
        {
            double delta=0;
            for(int j=0;j<class_label_number;j++)
            {
                delta+=responsibility1[j]*weight1[i][j];
            }
            double h1=hidden_layer_neurons1[i];
            double res=h1*(1-h1)*delta;
            responsibility2[i]=res;
            
        }
        //Output of Hidden Layer Neurons2(Responsibility3)
        for(int i=0;i<attribute_number;i++)
        {
            double delta=0;
            for(int j=0;j<attribute_number;j++)
            {
                delta+=responsibility2[j]*weight2[i][j];
            }
            double h2=hidden_layer_neurons2[i];
            double res=h2*(1-h2)*delta;
            responsibility3[i]=res;
        }
    }
    void weight_modification(example ex)
    {
        //Modification of Weight1
        for(int i=0;i<attribute_number;i++)
        {
            for(int j=0;j<class_label_number;j++)
            {
                double h1=hidden_layer_neurons1[i];
                weight1[i][j]+=ETA*responsibility1[j]*h1;
            }
        }
        //Modification of Weight2
        for(int i=0;i<attribute_number;i++)
        {
            for(int j=0;j<attribute_number;j++)
            {
                double h2=hidden_layer_neurons2[i];
                weight2[i][j]+=ETA*responsibility2[j]*h2;
            }
        }
        //Modification of Weight3
        for(int i=0;i<attribute_number;i++)
        {
            for(int j=0;j<attribute_number;j++)
            {
                double x=ex[i];
                weight3[i][j]+=ETA*responsibility3[j]*x;
            }
        }
    }
    void MSE_calculation(void)
    {
        MSE_past=MSE_now;
        double MSE=0;
        for(int i=0;i<class_label_number;i++)
        {
            MSE+=pow((target_vector[i]-output_layer_neuron[i]),2);
        }
        MSE_now=MSE/attribute_number;
        if(is_convergence())cout<<"->Convergence..........."<<endl;
    }
    bool is_convergence(void)
    {
        if((fabs(MSE_now-MSE_past)/MSE_past)<=pow(10, -4))return true;
        else return false;
    }
    void print_training_set(void)
    {
        cout<<"->Training Set:"<<endl;
        for(auto ex:training_set)
        {
            print_example(ex);
        }
    }
    void print_example(example ex)
    {
        cout<<"->Exmaple:";
        for(int i=0;i<ex.size();i++)
        {
            if(i!=attribute_number)cout<<ex[i]<<" ";
            else cout<<class_labels[ex[i]];
//            cout<<ex[i]<<" ";
        }
        cout<<endl;
    }
    void print_class_labels(void)
    {
        cout<<"->Class Labels:"<<endl;
        for(int i=0;i<class_labels.size();i++)
        {
            cout<<i<<": "<<class_labels[i]<<endl;
        }
    }
    void print_target_vector(example ex)
    {
        cout<<"->Target Vector:";
        for(int i=0;i<class_label_number;i++)
            cout<<target_vector[i]<<" ";
        cout<<endl;
    }
    void print_weights(void)
    {
        //print Weight1
        cout<<"->Weight:"<<endl;
        cout<<"================Weight1:======================================================"<<endl;
        for(int i=0;i<attribute_number;i++)
        {
            for(int j=0;j<class_label_number;j++)
            {
                cout<<"weight1("<<i<<","<<j<<"):"<<weight1[i][j]<<", ";
            }
            cout<<endl;
        }
        cout<<"================Weight2:======================================================"<<endl;

        for(int i=0;i<attribute_number;i++)
        {
            for(int j=0;j<attribute_number;j++)
            {
                cout<<"weight2("<<i<<","<<j<<"):"<<weight2[i][j]<<", ";
            }
            cout<<endl;
        }
        cout<<"================Weight3:======================================================"<<endl;

        for(int i=0;i<attribute_number;i++)
        {
            for(int j=0;j<attribute_number;j++)
            {
                cout<<"weight3("<<i<<","<<j<<"):"<<weight3[i][j]<<", ";
            }
            cout<<endl;
        }
    }
    void print_layer_neurons(void)
    {
        cout<<"->Layer Neurons......."<<endl;
        cout<<"Output Layer Neurons:"<<endl;
        for(int i=0;i<class_label_number;i++)
            cout<<"y("<<i<<"):"<<output_layer_neuron[i]<<" ";
        cout<<endl;
        cout<<"Hidden Layer Neurons1"<<endl;
        for(int i=0;i<attribute_number;i++)
            cout<<"h1("<<i<<"):"<<hidden_layer_neurons1[i]<<" ";
        cout<<endl;
        cout<<"Hidden Layer Neurons2"<<endl;
        for(int i=0;i<attribute_number;i++)
            cout<<"h2("<<i<<"):"<<hidden_layer_neurons2[i]<<" ";
        cout<<endl;
    }
    void print_output_signals(void)
    {
        cout<<"->Output Signals:";
        for(int i=0;i<class_label_number;i++)
        {
            cout<<"Y("<<i<<")="<<output_layer_neuron[i]<<" ";
        }
        cout<<endl;
    }
    void print_MSE(void)
    {
        cout<<"->MSE="<<MSE_now<<endl;
    }
};

int main(void)
{
    int attribute_number=4;
//    cin>>attribute_number;
    int instance_number=150;
//    cin>>instance_number;
    int class_label_number=3;
    Neural_Network Classifier(attribute_number, instance_number, class_label_number);
    int i=0;
    while (instance_number--)
    {
        example ex;
        string class_label;
        double attribute_value;
        cout<<i++<<": ";
        for(int i=0;i<attribute_number;i++)
        {
            cin>>attribute_value;
            ex.push_back(attribute_value);
            getchar();
        }
        cin>>class_label;
        Classifier.add_to_training_set(ex, class_label);//to build a Training Set
//        Classifier.Classification(ex, class_label);//to Classify and Modify weights
    }
    cout<<"->ETA="<<ETA<<endl;
    Classifier.print_weights();
//    Classifier.print_training_set();
//    Classifier.print_class_labels();
    return 0;
}

/*
 5.1,3.5,1.4,0.2,Iris-setosa
 4.9,3.0,1.4,0.2,Iris-setosa
 4.7,3.2,1.3,0.2,Iris-setosa
 4.6,3.1,1.5,0.2,Iris-setosa
 5.0,3.6,1.4,0.2,Iris-setosa
 5.4,3.9,1.7,0.4,Iris-setosa
 4.6,3.4,1.4,0.3,Iris-setosa
 5.0,3.4,1.5,0.2,Iris-setosa
 4.4,2.9,1.4,0.2,Iris-setosa
 4.9,3.1,1.5,0.1,Iris-setosa
 7.0,3.2,4.7,1.4,Iris-versicolor
 6.4,3.2,4.5,1.5,Iris-versicolor
 6.9,3.1,4.9,1.5,Iris-versicolor
 5.5,2.3,4.0,1.3,Iris-versicolor
 6.5,2.8,4.6,1.5,Iris-versicolor
 5.7,2.8,4.5,1.3,Iris-versicolor
 6.3,3.3,4.7,1.6,Iris-versicolor
 4.9,2.4,3.3,1.0,Iris-versicolor
 6.6,2.9,4.6,1.3,Iris-versicolor
 5.2,2.7,3.9,1.4,Iris-versicolor
 6.3,3.3,6.0,2.5,Iris-virginica
 5.8,2.7,5.1,1.9,Iris-virginica
 7.1,3.0,5.9,2.1,Iris-virginica
 6.3,2.9,5.6,1.8,Iris-virginica
 6.5,3.0,5.8,2.2,Iris-virginica
 7.6,3.0,6.6,2.1,Iris-virginica
 4.9,2.5,4.5,1.7,Iris-virginica
 7.3,2.9,6.3,1.8,Iris-virginica
 6.7,2.5,5.8,1.8,Iris-virginica
 7.2,3.6,6.1,2.5,Iris-virginica
 
 5.4,3.7,1.5,0.2,Iris-setosa
 4.8,3.4,1.6,0.2,Iris-setosa
 4.8,3.0,1.4,0.1,Iris-setosa
 4.3,3.0,1.1,0.1,Iris-setosa
 5.8,4.0,1.2,0.2,Iris-setosa
 5.7,4.4,1.5,0.4,Iris-setosa
 5.4,3.9,1.3,0.4,Iris-setosa
 5.1,3.5,1.4,0.3,Iris-setosa
 5.7,3.8,1.7,0.3,Iris-setosa
 5.1,3.8,1.5,0.3,Iris-setosa
 5.0,2.0,3.5,1.0,Iris-versicolor
 5.9,3.0,4.2,1.5,Iris-versicolor
 6.0,2.2,4.0,1.0,Iris-versicolor
 6.1,2.9,4.7,1.4,Iris-versicolor
 5.6,2.9,3.6,1.3,Iris-versicolor
 6.7,3.1,4.4,1.4,Iris-versicolor
 5.6,3.0,4.5,1.5,Iris-versicolor
 5.8,2.7,4.1,1.0,Iris-versicolor
 6.2,2.2,4.5,1.5,Iris-versicolor
 5.6,2.5,3.9,1.1,Iris-versicolor
 6.5,3.2,5.1,2.0,Iris-virginica
 6.4,2.7,5.3,1.9,Iris-virginica
 6.8,3.0,5.5,2.1,Iris-virginica
 5.7,2.5,5.0,2.0,Iris-virginica
 5.8,2.8,5.1,2.4,Iris-virginica
 6.4,3.2,5.3,2.3,Iris-virginica
 6.5,3.0,5.5,1.8,Iris-virginica
 7.7,3.8,6.7,2.2,Iris-virginica
 7.7,2.6,6.9,2.3,Iris-virginica
 6.0,2.2,5.0,1.5,Iris-virginica
 
 5.0,3.5,1.3,0.3,Iris-setosa
 4.5,2.3,1.3,0.3,Iris-setosa
 4.4,3.2,1.3,0.2,Iris-setosa
 5.0,3.5,1.6,0.6,Iris-setosa
 5.1,3.8,1.9,0.4,Iris-setosa
 4.8,3.0,1.4,0.3,Iris-setosa
 5.1,3.8,1.6,0.2,Iris-setosa
 4.6,3.2,1.4,0.2,Iris-setosa
 5.3,3.7,1.5,0.2,Iris-setosa
 5.0,3.3,1.4,0.2,Iris-setosa
 5.5,2.6,4.4,1.2,Iris-versicolor
 6.1,3.0,4.6,1.4,Iris-versicolor
 5.8,2.6,4.0,1.2,Iris-versicolor
 5.0,2.3,3.3,1.0,Iris-versicolor
 5.6,2.7,4.2,1.3,Iris-versicolor
 5.7,3.0,4.2,1.2,Iris-versicolor
 5.7,2.9,4.2,1.3,Iris-versicolor
 6.2,2.9,4.3,1.3,Iris-versicolor
 5.1,2.5,3.0,1.1,Iris-versicolor
 5.7,2.8,4.1,1.3,Iris-versicolor
 6.7,3.1,5.6,2.4,Iris-virginica
 6.9,3.1,5.1,2.3,Iris-virginica
 5.8,2.7,5.1,1.9,Iris-virginica
 6.8,3.2,5.9,2.3,Iris-virginica
 6.7,3.3,5.7,2.5,Iris-virginica
 6.7,3.0,5.2,2.3,Iris-virginica
 6.3,2.5,5.0,1.9,Iris-virginica
 6.5,3.0,5.2,2.0,Iris-virginica
 6.2,3.4,5.4,2.3,Iris-virginica
 5.9,3.0,5.1,1.8,Iris-virginica
 
 5.4,3.4,1.7,0.2,Iris-setosa
 5.1,3.7,1.5,0.4,Iris-setosa
 4.6,3.6,1.0,0.2,Iris-setosa
 5.1,3.3,1.7,0.5,Iris-setosa
 4.8,3.4,1.9,0.2,Iris-setosa
 5.0,3.0,1.6,0.2,Iris-setosa
 5.0,3.4,1.6,0.4,Iris-setosa
 5.2,3.5,1.5,0.2,Iris-setosa
 5.2,3.4,1.4,0.2,Iris-setosa
 4.7,3.2,1.6,0.2,Iris-setosa
 5.9,3.2,4.8,1.8,Iris-versicolor
 6.1,2.8,4.0,1.3,Iris-versicolor
 6.3,2.5,4.9,1.5,Iris-versicolor
 6.1,2.8,4.7,1.2,Iris-versicolor
 6.4,2.9,4.3,1.3,Iris-versicolor
 6.6,3.0,4.4,1.4,Iris-versicolor
 6.8,2.8,4.8,1.4,Iris-versicolor
 6.7,3.0,5.0,1.7,Iris-versicolor
 6.0,2.9,4.5,1.5,Iris-versicolor
 5.7,2.6,3.5,1.0,Iris-versicolor
 6.9,3.2,5.7,2.3,Iris-virginica
 5.6,2.8,4.9,2.0,Iris-virginica
 7.7,2.8,6.7,2.0,Iris-virginica
 6.3,2.7,4.9,1.8,Iris-virginica
 6.7,3.3,5.7,2.1,Iris-virginica
 7.2,3.2,6.0,1.8,Iris-virginica
 6.2,2.8,4.8,1.8,Iris-virginica
 6.1,3.0,4.9,1.8,Iris-virginica
 6.4,2.8,5.6,2.1,Iris-virginica
 7.2,3.0,5.8,1.6,Iris-virginica
 
 4.8,3.1,1.6,0.2,Iris-setosa
 5.4,3.4,1.5,0.4,Iris-setosa
 5.2,4.1,1.5,0.1,Iris-setosa
 5.5,4.2,1.4,0.2,Iris-setosa
 4.9,3.1,1.5,0.1,Iris-setosa
 5.0,3.2,1.2,0.2,Iris-setosa
 5.5,3.5,1.3,0.2,Iris-setosa
 4.9,3.1,1.5,0.1,Iris-setosa
 4.4,3.0,1.3,0.2,Iris-setosa
 5.1,3.4,1.5,0.2,Iris-setosa
 5.5,2.4,3.8,1.1,Iris-versicolor
 5.5,2.4,3.7,1.0,Iris-versicolor
 5.8,2.7,3.9,1.2,Iris-versicolor
 6.0,3.4,4.5,1.6,Iris-versicolor
 6.7,3.1,4.7,1.5,Iris-versicolor
 5.6,3.0,4.1,1.3,Iris-versicolor
 5.5,2.5,4.0,1.3,Iris-versicolor
 7.4,2.8,6.1,1.9,Iris-virginica
 7.9,3.8,6.4,2.0,Iris-virginica
 6.4,2.8,5.6,2.2,Iris-virginica
 6.3,2.8,5.1,1.5,Iris-virginica
 6.1,2.6,5.6,1.4,Iris-virginica
 7.7,3.0,6.1,2.3,Iris-virginica
 6.3,3.4,5.6,2.4,Iris-virginica
 6.4,3.1,5.5,1.8,Iris-virginica
 6.0,3.0,4.8,1.8,Iris-virginica
 6.9,3.1,5.4,2.1,Iris-virginica
*/









