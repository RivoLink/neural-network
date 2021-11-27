package mg.rivolink.ai;

public class Network{

	public final float alpha=0.1f;

	public final int i;
	public final Layer hidL;
	public final Layer outL;

	public Network(int i,int j,int k){
		this.i=i;
		this.hidL=new Layer(i,j);
		this.outL=new Layer(j,k);
	}
	
	public float[] predict(int bits){
		return setInput(bits).getOutputs();
	}

	public float[] predict(float... inputs){
		return setInput(inputs).getOutputs();
	}

	public void train(int bits,int[] target){
		setInput(bits);
		training(target);
	}

	public void train(float[] inputs,int[] target){
		setInput(inputs);
		training(target);
	}
	
	public void train(int[] bitsArray,int[][] ytrains,int epochs){
		int size=Math.min(bitsArray.length,ytrains.length);
		for(int epoch=0;epoch<epochs;epoch++){
			for(int i=0;i<size;i++){
				this.train(bitsArray[i],ytrains[i]);
			}
		}
	}
	
	public void train(float[][] xtrains,int[][] ytrains,int epochs){
		int size=Math.min(xtrains.length,ytrains.length);
		for(int epoch=0;epoch<epochs;epoch++){
			for(int i=0;i<size;i++){
				this.train(xtrains[i],ytrains[i]);
			}
		}
	}

	private Network setInput(int bits){
		hidL.setInputs(bits);
		return this;
	}

	private Network setInput(float... inputs){
		hidL.setInputs(inputs);
		return this;
	}

	private float[] getOutputs(){
		float[] x=hidL.getOutputs();
		return outL.setInputs(x).getOutputs();
	}
	
	private void training(int[] target){
		float[] yhat=getOutputs();
		float[] x=hidL.getOutputs();
		int[] y=target;

		Neuron[] hN=hidL.neurons;
		Neuron[] oN=outL.neurons;

		float gE_wij,gE_wjk;
		for(int j=0;j<hN.length;j++){
			for(int k=0;k<oN.length;k++){
				gE_wjk=alpha*(y[k]-yhat[k])*yhat[k]*(1-yhat[k]);
				oN[k].biais+=gE_wjk;
				oN[k].weights[j]+=gE_wjk*x[j];
			}

			for(int i=0;i<this.i;i++){
				gE_wij=0;
				float xi=hidL.neurons[0].inputs[i];
				for(int k=0;k<oN.length;k++){
					float wjk=outL.neurons[k].weights[j];
					gE_wij+=(y[k]-yhat[k])*yhat[k]*(1-yhat[k])*wjk*x[j]*(1-x[j]);
				}
				gE_wij*=alpha;
				hN[j].biais+=gE_wij;
				hN[j].weights[i]+=gE_wij*xi;
			}
		}
	}

}
