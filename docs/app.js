import { $, bars, diverge, fail, int, kpis, load, num, pct, seg } from './kit.js';

try {
  const d = await load();
  const S = d.strategies, names = Object.keys(S);
  kpis($('#kpis'), [
    { label: 'Failures captured, same budget', value: `${num(S.boosted.failures, 1)} vs ${num(S.uniform.failures, 1)}`, note: 'boosted against uniform, of 200 labelled cases' },
    { label: 'Outliers fail more', value: pct(d.failure_rate_outliers), note: `of the model's answers to outliers are wrong; ${pct(d.failure_rate_clustered)} inside clusters` },
    { label: 'Accuracy estimate error', value: `${num(100 * S.boosted.rmse, 2)} pts`, note: `RMSE with boosted sampling; uniform ${num(100 * S.uniform.rmse, 2)}` },
    { label: 'Production model accuracy', value: pct(d.model_accuracy), note: `on all ${int(d.candidates)} candidates, the number the set must estimate` },
  ]);
  bars($('#funnel'), [
    { label: 'logged messages', value: d.logs, text: int(d.logs), color: 'var(--c6)' },
    { label: `deduplicated (−${int(d.near_duplicates_dropped)})`, title: `${int(d.near_duplicates_dropped)} near-duplicates dropped`, value: d.candidates, text: int(d.candidates) },
    { label: `in ${d.clusters} clusters`, value: d.candidates - d.outliers, text: int(d.candidates - d.outliers) },
    { label: 'outliers', value: d.outliers, text: int(d.outliers), color: 'var(--c5)' },
    { label: 'labelled', value: d.budget, text: String(d.budget), color: 'var(--good)' },
  ], { max: d.logs });
  $('#clusters').textContent = `Largest clusters, named by their top terms: ${d.cluster_names.join('; ')}. Set version ${d.version}: the same logs and settings always build the same set.`;

  const M = {
    failures: ['Model failures captured', (s) => s.failures, (v) => num(v, 1), 'wrong answers in the set: the cases an eval exists to find'],
    intents: ['Intents covered', (s) => s.intents, (v) => num(v, 1), `of ${d.intents} intents in the traffic`],
    clusters: ['Clusters covered', (s) => s.clusters, (v) => num(v, 1), `of ${d.clusters} clusters`],
    rmse: ['Accuracy estimate RMSE (pts)', (s) => 100 * s.rmse, (v) => num(v, 2), 'lower is better: how far the weighted accuracy on the set lands from the true accuracy'],
  };
  seg($('#metric'), Object.entries(M).map(([k, m]) => [k, m[0]]), 'failures', (k) => {
    const [, get, fmt, note] = M[k];
    bars($('#bars'), names.map((n) => ({ label: n, value: get(S[n]), text: fmt(get(S[n])), color: n === 'boosted' ? 'var(--accent)' : 'var(--c6)' })));
    $('#metricNote').textContent = note;
  });
  diverge($('#bias'), names.flatMap((n) => [
    { label: `${n} · weighted`, value: 100 * S[n].bias, text: `${num(100 * S[n].bias, 2)}`, color: 'var(--accent)' },
    { label: `${n} · plain average`, value: 100 * S[n].naive_bias, text: `${num(100 * S[n].naive_bias, 2)}`, color: Math.abs(S[n].naive_bias) > 0.01 ? 'var(--bad)' : 'var(--c6)' },
  ]), { max: 6 });
} catch (err) {
  fail(err);
}
