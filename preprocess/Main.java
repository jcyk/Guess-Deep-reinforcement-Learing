import edu.stanford.nlp.process.*;
import java.io.*;
public class Main{
  public static void main(String[] args){
  		try{
  			File file = new File(args[0]);
  			BufferedReader reader = new BufferedReader(new FileReader(file));
  			BufferedWriter writer = new BufferedWriter(new FileWriter("stemmed"));
			Stemmer s = new Stemmer();
			String line = null;
			while ((line = reader.readLine())!=null){
				writer.write(s.stem(line));
				writer.newLine();
			}
			reader.close();
			writer.close();
		} catch (IOException e){
			e.printStackTrace();
		}
	}
}
