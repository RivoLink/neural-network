import java.util.*;
import mg.rivolink.ai.*;

public class Main{
	
	static int[][] targets={
		{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{0,1,0},{0,1,0},{0,1,0},
		{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},
		{1,0,0},{1,0,0},{1,0,0},{1,0,0},{0,0,1},{1,0,0},{1,0,0},{1,0,0},
		{0,0,1},{0,1,0},{0,1,0},{0,1,0},{0,0,1},{0,1,0},{0,1,0},{0,1,0},
		{1,0,0},{1,0,0},{1,0,0},{1,0,0},{0,0,1},{1,0,0},{1,0,0},{1,0,0},
		{0,0,1},{0,0,1},{0,1,0},{0,1,0},{0,0,1},{0,0,1},{0,1,0},{0,1,0},
		{1,0,0},{1,0,0},{1,0,0},{1,0,0},{0,0,1},{1,0,0},{1,0,0},{1,0,0},
		{0,0,1},{0,0,1},{0,1,0},{0,1,0},{0,0,1},{0,0,1},{0,1,0},{0,1,0},
	};
	
	public static void main(String[] args){
		Network network=new Network(6,4,3);
		
		for(int epoch=0;epoch<100;epoch++){
			for(int input=0;input<targets.length;input++){
				int[] target=targets[input];
				network.train(input,target);
			}
		}
		
		printr(network.predict(0b110011));
		printr(network.predict(new float[]{1,1,0,0,1,1}));
	}
	
	static void printr(float[] array){
		System.out.print("[");
		
		for(int i=0;i<array.length;i++){
			System.out.print(array[i]);
			
			if(i<array.length-1)
				System.out.print(",");
		}
		
		System.out.println("]");
	}
}
