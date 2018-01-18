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

import tud.ke.ml.project.util.Pair;

/**
 * This implementation assumes the class attribute is always available (but
 * probably not set).
 * 
 */
public class NearestNeighbor extends INearestNeighbor implements Serializable {
	private static final long serialVersionUID = 1L;

	private List<List<Object>> model;
	private List<Object> testinstance;

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

		for (Pair<List<Object>, Double> instance : subset) {
			Double d = vote.putIfAbsent(instance.getA().get(getClassAttribute()), instance.getB());

			if (d != null) {
				vote.put(instance.getA().get(getClassAttribute()), instance.getB() + d);
			}
		}
		return vote;
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> vote = new HashMap<Object, Double>();

		for (Pair<List<Object>, Double> instance : subset) {
			Double d = vote.putIfAbsent(instance.getA().get(getClassAttribute()), Math.pow(1 / instance.getB(), 1));

			if (d != null) {
				vote.put(instance.getA().get(getClassAttribute()), Math.pow((1 / instance.getB()), 1) + d);
			}
		}

		return vote;
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
		Object result = null;

		if (isInverseWeighting()) {
			result = getWinner(getWeightedVotes(subset));
		} else {
			result = getWinner(getUnweightedVotes(subset));
		}

		return result;
	}

	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> data) {
		List<Pair<List<Object>, Double>> instances = new ArrayList<Pair<List<Object>, Double>>();
		
		List<List<Object>> normalized = new ArrayList<>(this.model);

		// Normalization
		if (isNormalizing()) {
			this.testinstance = data;

			double[][] norm = normalizationScaling();
			this.scaling = norm[0];
			this.translation = norm[1];

			for (int i = 0; i < data.size(); i++) {
				Object elem = data.get(i);

				if (elem instanceof String) {
					continue;
				} // Early escape

				data.set(i, ((Double) elem - this.translation[i]) / (this.scaling[i]));
			}

			for (List<Object> list : normalized) {
				for (int i = 0; i < list.size(); i++) {
					Object elem = list.get(i);

					if (elem instanceof String) {
						continue; // Early escape
					}

					list.set(i, ((Double) elem - this.translation[i]) / (this.scaling[i]));
				}
			}
			
		}

		if (getMetric() == 0) {
			for (List<Object> instance : normalized) {
				Double d = determineManhattanDistance(instance, data);
				Pair<List<Object>, Double> p = new Pair<List<Object>, Double>(instance, d);
				instances.add(p);
			}
		} else {
			for (List<Object> instance : normalized) {
				Double d = determineEuclideanDistance(instance, data);
				Pair<List<Object>, Double> p = new Pair<List<Object>, Double>(instance, d);
				instances.add(p);
			}
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

		for (int i = 0; i < instance1.size() - 1; i++) {
			if (instance1.get(i) instanceof Double && instance2.get(i) instanceof Double) {
				distance += Math.abs((Double) instance1.get(i) - (Double) instance2.get(i));
			} else if (instance1.get(i) instanceof String && instance2.get(i) instanceof String) {
				if (!instance1.get(i).equals(instance2.get(i))) {
					distance += 1;
				}
			}
		}

		return distance;
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {
		double distance = 0.0;

		for (int i = 0; i < instance1.size() - 1; i++) {
			if (instance1.get(i) instanceof Double && instance2.get(i) instanceof Double) {
				distance += Math.pow((Double) instance1.get(i) - (Double) instance2.get(i), 2);
			} else if (instance1.get(i) instanceof String && instance2.get(i) instanceof String) {
				if (!instance1.get(i).equals(instance2.get(i))) {
					distance += 1;
				}
			}
		}

		return Math.sqrt(distance);
	}

	@Override
	protected double[][] normalizationScaling() {
		scaling = new double[this.testinstance.size()];
		translation = new double[this.testinstance.size()];

		double[] min = new double[this.testinstance.size()];
		double[] max = new double[this.testinstance.size()];

		for (int i = 0; i < min.length; i++) {
			min[i] = Double.MAX_VALUE;
			max[i] = Double.MIN_VALUE;
			translation[i] = Double.MAX_VALUE;
		}

		List<List<Object>> merged = new ArrayList<>(this.model);
		merged.add(testinstance);

		for (List<Object> list : merged) {
			for (int i = 0; i < list.size(); i++) {
				Object elem = list.get(i);

				if (elem instanceof String) {
					translation[i] = 0;
					min[i] = 0;
					max[i] = 1;
					continue;
				}

				// elem is a Double
				Double d = (Double) elem;
				if (d < translation[i]) {
					translation[i] = d;
				}

				if (d < min[i]) {
					min[i] = d;
				}
				if (d > max[i]) {
					max[i] = d;
				}
			}
		}

		for (int i = 0; i < scaling.length; i++) {
			//max-translation because we divide by the translated max, so we should save it.
			scaling[i] = Math.abs(max[i] - min[i]);
		}

		return new double[][] { scaling, translation };
	}
	
	/*
	@Override
	protected double[][] normalizationScaling() {
		double[][] factors = new double[model.size()][model.get(0).size()];
		
		if(isNormalizing()) {
			double[] min = new double[model.get(0).size()];
			double[] max = new double[model.get(0).size()];
			
			for(int i = 0; i < model.size(); i++) {
				List<Object> instance = model.get(i);
				
				for(int j = 0; j < instance.size(); j++) {
					Object attribute = instance.get(j);
					
					if(attribute instanceof Double) {
						double val = ((Double) attribute).doubleValue();
						
						if(min[j] > val) {
							min[j] = val;
						}
						
						if(max[j] < val) {
							max[j] = val;
						}
					}
				}
			}
			
			for(int i = 0; i < model.size(); i++) {
				List<Object> instance = model.get(i);
				
				for(int j = 0; j < instance.size(); j++) {
					Object attribute = instance.get(j);
					
					if(attribute instanceof String) {
						factors[i][j] = 1.0;
					}else {
						factors[i][j] = (((Double) attribute).doubleValue() - min[j]) / (max[j] + min[j]);
					}
				}
			}
			
			return factors;
		}
		
		return factors;
	}
	*/

	public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
		return map.entrySet().stream().sorted(Map.Entry.comparingByValue(Collections.reverseOrder()))
				.collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));
	}

}
