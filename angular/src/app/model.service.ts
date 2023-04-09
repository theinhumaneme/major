import { HttpClient } from '@angular/common/http';
import { ComponentFactoryResolver, Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class ModelService {
  resp: any;
  constructor(private http: HttpClient) {}

  predict_model(values: any) {
    let model_vals = [];
    model_vals.push(values['baseline_value']);
    model_vals.push(values['accelerations']);
    model_vals.push(values['fetal_movement']);
    model_vals.push(values['uterine_contractions']);
    model_vals.push(values['light_decelerations']);
    model_vals.push(values['severe_decelerations']);
    model_vals.push(values['prolongued_decelerations']);
    model_vals.push(values['abnormal_short_term_variability']);
    model_vals.push(values['mean_value_of_short_term_variability']);
    model_vals.push(
      values['percentage_of_time_with_abnormal_long_term_variability']
    );
    model_vals.push(values['mean_value_of_long_term_variability']);
    model_vals.push(values['histogram_width']);
    model_vals.push(values['histogram_min']);
    model_vals.push(values['histogram_max']);
    model_vals.push(values['histogram_number_of_peaks']);
    model_vals.push(values['histogram_number_of_zeroes']);
    model_vals.push(values['histogram_mode']);
    model_vals.push(values['histogram_mean']);
    model_vals.push(values['histogram_median']);
    model_vals.push(values['histogram_variance']);
    model_vals.push(values['histogram_tendency']);
    let sub = this.http.post('http://10.5.0.4:5000/opt', {
      values: model_vals,
      model: values['selected_model'],
    });
    return sub;
  }

  pred_model_array(values, model) {
    let sub = this.http.post('http://10.5.0.4:5000/opt', {
      values: values,
      model: model,
    });
    return sub;
  }
}
