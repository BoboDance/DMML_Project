package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Map.Entry;

import tud.ke.ml.project.util.Pair;

/**
 * This implementation assumes the class attribute is always available (but
 * probably not set).
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

		for (Pair<List<Object>, Double> instance : subset) {
			Double d = vote.putIfAbsent(instance.getA().get(getClassAttribute()), 1.0);

			if (d != null) {
				vote.put(instance.getA().get(getClassAttribute()), 1.0 + d);
			}
		}
		return vote;
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> vote = new HashMap<Object, Double>();

		for (Pair<List<Object>, Double> instance : subset) {
			Double d = vote.putIfAbsent(instance.getA().get(getClassAttribute()), 1 / instance.getB());

			if (d != null) {
				vote.put(instance.getA().get(getClassAttribute()), (1 / instance.getB()) + d);
			}
		}

		return vote;
	}

	@Override
	protected Object getWinner(Map<Object, Double> votes) {

		Object win = null;
		double max = Double.MIN_VALUE;

		for (Entry<Object, Double> instance : votes.entrySet()) {
			if (instance.getValue() > max) {
				win = instance.getKey();
				max = instance.getValue();
			}
		}

		return win;
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
		ListIterator<List<Object>> iter = this.model.listIterator();

		if (isNormalizing()) {

			double[][] norm = normalizationScaling();

			this.scaling = norm[0];
			this.translation = norm[1];
		}

		while (iter.hasNext()) {

			List<Object> data_clone = (LinkedList<Object>) ((LinkedList<Object>) data).clone();
			List<Object> model_clone = (LinkedList<Object>) ((LinkedList<Object>) iter.next()).clone();

			if (isNormalizing()) {

				for (int i = 0; i < data_clone.size(); i++) {
					Object attr1 = data_clone.get(i);
					Object attr2 = model_clone.get(i);

					if (attr1 instanceof Double && attr2 instanceof Double) {
						data_clone.set(i, ((Double) attr1 - translation[i]) / (scaling[i] + 0.000001));
						model_clone.set(i, ((Double) attr2 - translation[i]) / (scaling[i] + 0.000001));
					}
				}
			}

			Double distance;

			if (getMetric() == 0) {
				distance = determineManhattanDistance(data_clone, model_clone);

			} else {
				distance = determineEuclideanDistance(data_clone, model_clone);
			}

			instances.add(new Pair<List<Object>, Double>(model_clone, distance));
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

		for (int i = 0; i < instance1.size(); i++) {
			if (i == this.getClassAttribute())
				continue;
			if (instance1.get(i) instanceof Double && instance2.get(i) instanceof Double) {
				distance += Math.abs((Double) instance1.get(i) - (Double) instance2.get(i));
			} else if (instance1.get(i) instanceof String && instance2.get(i) instanceof String) {
				if (!instance1.get(i).equals(instance2.get(i))) {
					distance += 1.0;
				}
			}
		}

		return distance;
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {
		double distance = 0.0;

		for (int i = 0; i < instance1.size(); i++) {
			if (i == this.getClassAttribute())
				continue;
			if (instance1.get(i) instanceof Double && instance2.get(i) instanceof Double) {
				distance += Math.pow((Double) instance1.get(i) - (Double) instance2.get(i), 2);
			} else if (instance1.get(i) instanceof String && instance2.get(i) instanceof String) {
				if (!instance1.get(i).equals(instance2.get(i))) {
					distance += 1.0;
				}
			}
		}

		return Math.sqrt(distance);
	}

	@Override
	protected double[][] normalizationScaling() {

		scaling = new double[this.model.get(0).size()];

		double[] min = new double[this.model.get(0).size()];
		double[] max = new double[this.model.get(0).size()];

		for (int i = 0; i < min.length; i++) {
			max[i] = Double.MIN_VALUE;
			min[i] = Double.MAX_VALUE;
		}

		List<List<Object>> merged = new ArrayList<>(this.model);
		// merged.add(testinstance);

		for (List<Object> list : merged) {
			for (int i = 0; i < list.size(); i++) {
				Object elem = list.get(i);

				if (elem instanceof String) {
					continue;
				}

				// elem is a Double
				Double d = (Double) elem;
				if (d < min[i]) {
					min[i] = d;
				}

				if (d > max[i]) {
					max[i] = d;
				}
			}
		}

		for (int i = 0; i < scaling.length; i++) {
			// max-translation because we divide by the translated max, so we should save
			// it.
			scaling[i] = max[i] - min[i];
			// scaling[i] = 1;
			// translation[i] = 0;
		}

		return new double[][] { scaling, min };
	}
}