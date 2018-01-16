package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import tud.ke.ml.project.util.Pair;

/**
 * This implementation assumes the class attribute is always available (but probably not set).
 * 
 */
public class NearestNeighbor extends INearestNeighbor implements Serializable {
	private static final long serialVersionUID = 1L;

	private List<List<Object>> model;
	
	protected double[] scaling;
	protected double[] translation;

	@Override
	public String getMatrikelNumbers() {
		return "2629955,2493301,2792549";
	}

	@Override
	protected void learnModel(List<List<Object>> data) {
		this.model = data;
	}

	@Override
	protected Map<Object, Double> getUnweightedVotes(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> vote = new HashMap<Object, Double>();
		
		for(Pair<List<Object>, Double> instance : subset) {
			vote.put(instance.getA(), vote.get(instance.getA()) == null ? instance.getB() : vote.get(instance.getA()) + instance.getB());
		}
		
		return vote;
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {
		throw new NotImplementedException();
	}

	@Override
	protected Object getWinner(Map<Object, Double> votes) {
		votes = sortByValue(votes);
		
		Map.Entry<Object, Double> entry = votes.entrySet().iterator().next();
		Object winner = entry.getKey();
		
		return winner;
	}

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		return getWinner(getUnweightedVotes(subset));
	}

	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> data) {
		List<Pair<List<Object>, Double>> instances = new ArrayList<Pair<List<Object>, Double>>();
		
		for(List<Object> instance : this.model) {
			Double d = determineManhattanDistance(instance, data);
			Pair<List<Object>, Double> p = new Pair<List<Object>, Double>(instance, d);
			instances.add(p);
		}
		
		Collections.sort(instances, new Comparator<Pair<List<Object>, Double>>() {
			@Override
			public int compare(Pair<List<Object>, Double> o1, Pair<List<Object>, Double> o2) {
				Double d1 = o1.getB();
				Double d2 = o2.getB();
				
				return d1.compareTo(d2);
			}
		});
		
		List<Pair<List<Object>, Double>> nearest = instances.subList(0, getkNearest());
		
		return nearest;
	}

	@Override
	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
		double distance = 0.0;
		
		for(int i=0; i < instance1.size(); i++) {
			if(instance1.get(i) instanceof Double && instance2.get(i) instanceof Double) {
				distance += Math.abs((Double) instance1.get(i) - (Double) instance2.get(i));	
			}
			else if(instance1.get(i) instanceof String && instance2.get(i) instanceof String) {
				if(! instance1.get(i).equals(instance2.get(i))) {
					distance += 1;
				}
			}
		}
		
		return distance;
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {
		throw new NotImplementedException();
	}

	@Override
	protected double[][] normalizationScaling() {
		throw new NotImplementedException();
	}
	
	public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
	    return map.entrySet()
	              .stream()
	              .sorted(Map.Entry.comparingByValue())
	              .collect(Collectors.toMap(
		               Map.Entry::getKey, 
		               Map.Entry::getValue, 
		               (e1, e2) -> e1, 
		               LinkedHashMap::new
	              ));
	}
	
}
 