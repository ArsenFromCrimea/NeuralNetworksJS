initialization=function(){
    var amp=10;
    for(var layer=0;layer<layers.length-1;layer++){
	var nextLayer=layers[layer+1];
	for(var knot=0;knot<layers[layer].length;knot++){
	    for(var nextKnot=0;nextKnot<nextLayer.length;nextKnot++){
		layers[layer][knot].toOutput[nextKnot]=(Math.random()-0.5)*amp;
	    }
	}
    }
    epoche=0;
    batch=batch_field.value;
    reset();
    countOfIterations=0;
}

layers=[new Array(2),
    new Array(8),
    new Array(18),
    new Array(1)
];

lastLayer=layers[layers.length-1];

for(layer=0;layer<layers.length;layer++){
    for(knot=0;knot<layers[layer].length;knot++){
	layers[layer][knot]={};
	if(layer<layers.length-1){
	    layers[layer][knot].toOutput=new Array(layers[layer+1].length);
	}
    }
}

approximation=function(x){
    for(layer=0;layer<layers.length;layer++){
	if(layer>0){
	    var previousLayer=layers[layer-1];
	}
	for(knot=0;knot<layers[layer].length;knot++){
	    if(layer==0){
		layers[layer][knot].hiddenValue=x[knot];
		layers[layer][knot].value=x[knot];
	    }else{
		layers[layer][knot].hiddenValue=0;
		for(var previousKnot=0;previousKnot<previousLayer.length;previousKnot++){
		    layers[layer][knot].hiddenValue+=
			previousLayer[previousKnot].value*
			previousLayer[previousKnot].toOutput[knot];
		}
		if(layer<layers.length-1){
		    layers[layer][knot].value=
			sigmoid(layers[layer][knot].hiddenValue);
		    layers[layer][knot].sigmoid1=
			sigmoid1(layers[layer][knot].hiddenValue);
		}else{
		    layers[layer][knot].value=
			layers[layer][knot].hiddenValue;
		    layers[layer][knot].sigmoid1=
			1;
		}
	    }
	}
    }
}

ourFunction=function(x){
    result=new Array(lastLayer.length);
    result[0]=Math.sin(x[0]);
    return result;
}

findMinAndMax=function(){
	for(var e=0;e<examples.length;e++){
	    if(e==0){
		min=examples[e].y[0];
		max=examples[e].y[0];
	    }else{
		if(examples[e].y[0]<min){
    		    min=examples[e].y[0];
		}else{
    		    if(examples[e].y[0]>max){
			max=examples[e].y[0];
    		    }
		}
	    }
	}
}

fillExamples=function(){
	var dx=(b-a)/(examples.length-1);
	for(var e=0;e<examples.length;e++){
	    examples[e]={};
	    examples[e].x=[a+e*dx,b/2];
	    examples[e].y=ourFunction(examples[e].x);
	}
}

y=new Array(lastLayer.length);

reset=function(){
    for(var i=0;i<examples.length;i++){
	examples[i].free=true;
    }
    rest=examples.length;
}

getLoss=function(){
    var max=0;
    var e_;
    errors=new Array(lastLayer.length);
    var findError=function(){
	fullError=0
	for(var knot=0;knot<lastLayer.length;knot++){
	    fullError+=Math.pow(errors[knot],2);
	}
    }
    for(var e=0;e<batchExamples.length;e++){
	approximation(batchExamples[e].x);
	for(var knot=0;knot<lastLayer.length;knot++){
	    errors[knot]=batchExamples[e].y[knot]-lastLayer[knot].value;
	}
	findError();
	if(fullError>max){
	    e_=e;
	    max=fullError;
	}
    }
    approximation(batchExamples[e_].x);
    for(var knot=0;knot<lastLayer.length;knot++){
	lastLayer[knot].error=errors[knot];
    }
}

iteration=function(){
    batchExamples=[];
    for(var c=0;c<batch_field.value&&rest>0;c++){
	index=Math.floor(Math.random()*rest);
	var j=-1;
	for(var i=0;i<examples.length&&j<index;i++){
	    if(examples[i].free){
		j++;
	    }
	}
	var l=batchExamples.length;
	batchExamples.length++;
	batchExamples[l]=examples[j];
	examples[j].free=false;
	rest--;
    }
    getLoss();
    for(var layer=layers.length-2;layer>0;layer--){
	var nextLayer=layers[layer+1];
	for(var knot=0;knot<layers[layer].length;knot++){
	    layers[layer][knot].error=0;
	    for(var nextKnot=0;nextKnot<nextLayer.length;nextKnot++){
		layers[layer][knot].error+=
		    nextLayer[nextKnot].error*layers[layer][knot].toOutput[nextKnot]*nextLayer[nextKnot].sigmoid1
	    }
	}
    }
    // change weights:
    for(var layer=0;layer<layers.length-1;layer++){
	var nextLayer=layers[layer+1];
	for(var knot=0;knot<layers[layer].length;knot++){
	    for(var nextKnot=0;nextKnot<nextLayer.length;nextKnot++){
		layers[layer][knot].toOutput[nextKnot]+=
		    eta*nextLayer[nextKnot].error*
		    nextLayer[nextKnot].sigmoid1*
		    layers[layer][knot].value;
	    }
	}
    }

    if(rest>0){
	return;
    }
    reset();
    epoche++;
    accuracy_field.innerHTML=Math.abs(lastLayer[0].error);
    countOfIterations_field.innerHTML=epoche;
}