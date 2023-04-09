import { Component, OnInit } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { ModelService } from '../model.service';

interface Model {
  model: string;
  viewValue: string;
}

@Component({
  selector: 'app-form',
  templateUrl: './form.component.html',
  styleUrls: ['./form.component.css'],
})
export class FormComponent implements OnInit {
  postForm: FormGroup;
  response: any;
  models: Model[] = [
    {
      model: 'GradientBoostingClassifierModelHyperTuned',
      viewValue: 'Optimal Model',
    },
    { model: 'LinearRegressionModel', viewValue: 'Linear Regression' },
    {
      model: 'DecisionTreeClassifierModel',
      viewValue: 'Decision Tree Classifier',
    },
    {
      model: 'GradientBoostingClassifierModel',
      viewValue: 'Gradient Boosting Classifier',
    },
    { model: 'KNeighborsClassifierModel', viewValue: 'K Nearest Neighbors' },
    {
      model: 'RandomForestClassifierModel',
      viewValue: 'Random Forest Classifer',
    },
    { model: 'SVCModel', viewValue: 'Support Vector Machine' },
  ];
  constructor(private modelService: ModelService) {
    this.initForm();
  }

  private initForm() {
    let baseline_value: Number = 0;
    let accelerations: Number = 0;
    let fetal_movement: Number = 0;
    let uterine_contractions: Number = 0;
    let light_decelerations: Number = 0;
    let severe_decelerations: Number = 0;
    let prolongued_decelerations: Number = 0;
    let abnormal_short_term_variability: Number = 0;
    let mean_value_of_short_term_variability: Number = 0;
    let percentage_of_time_with_abnormal_long_term_variability: Number = 0;
    let mean_value_of_long_term_variability: Number = 0;
    let histogram_width: Number = 0;
    let histogram_min: Number = 0;
    let histogram_max: Number = 0;
    let histogram_number_of_peaks: Number = 0;
    let histogram_number_of_zeroes: Number = 0;
    let histogram_mode: Number = 0;
    let histogram_mean: Number = 0;
    let histogram_median: Number = 0;
    let histogram_variance: Number = 0;
    let histogram_tendency: Number = 0;

    // let baseline_value: Number;
    // let accelerations: Number;
    // let fetal_movement: Number;
    // let uterine_contractions: Number;
    // let light_decelerations: Number;
    // let severe_decelerations: Number;
    // let prolongued_decelerations: Number;
    // let abnormal_short_term_variability: Number;
    // let mean_value_of_short_term_variability: Number;
    // let percentage_of_time_with_abnormal_long_term_variability: Number;
    // let mean_value_of_long_term_variability: Number;
    // let histogram_width: Number;
    // let histogram_min: Number;
    // let histogram_max: Number;
    // let histogram_number_of_peaks: Number;
    // let histogram_number_of_zeroes: Number;
    // let histogram_mode: Number;
    // let histogram_mean: Number;
    // let histogram_median: Number;
    // let histogram_variance: Number;
    // let histogram_tendency: Number;

    this.postForm = new FormGroup({
      baseline_value: new FormControl(baseline_value, Validators.required),
      accelerations: new FormControl(accelerations, Validators.required),
      fetal_movement: new FormControl(fetal_movement, Validators.required),
      uterine_contractions: new FormControl(
        uterine_contractions,
        Validators.required
      ),
      light_decelerations: new FormControl(
        light_decelerations,
        Validators.required
      ),
      severe_decelerations: new FormControl(
        severe_decelerations,
        Validators.required
      ),
      prolongued_decelerations: new FormControl(
        prolongued_decelerations,
        Validators.required
      ),
      abnormal_short_term_variability: new FormControl(
        abnormal_short_term_variability,
        Validators.required
      ),
      mean_value_of_short_term_variability: new FormControl(
        mean_value_of_short_term_variability,
        Validators.required
      ),
      percentage_of_time_with_abnormal_long_term_variability: new FormControl(
        percentage_of_time_with_abnormal_long_term_variability,
        Validators.required
      ),
      mean_value_of_long_term_variability: new FormControl(
        mean_value_of_long_term_variability,
        Validators.required
      ),
      histogram_width: new FormControl(histogram_width, Validators.required),
      histogram_min: new FormControl(histogram_min, Validators.required),
      histogram_max: new FormControl(histogram_max, Validators.required),
      histogram_number_of_peaks: new FormControl(
        histogram_number_of_peaks,
        Validators.required
      ),
      histogram_number_of_zeroes: new FormControl(
        histogram_number_of_zeroes,
        Validators.required
      ),
      histogram_mode: new FormControl(histogram_mode, Validators.required),
      histogram_mean: new FormControl(histogram_mean, Validators.required),
      histogram_median: new FormControl(histogram_median, Validators.required),
      histogram_variance: new FormControl(
        histogram_variance,
        Validators.required
      ),
      histogram_tendency: new FormControl(
        histogram_tendency,
        Validators.required
      ),
      selected_model: new FormControl('', Validators.required),
    });
    this.postForm.controls['selected_model'].setValue(this.models[0].model, {
      onlySelf: true,
    });
  }
  ngOnInit(): void {
    this.initForm();
  }
  onSubmit(): void {
    if (this.postForm.valid == true) {
      let val = this.modelService.predict_model(this.postForm.value)
      .subscribe((resp) => {
        this.response = resp;
      });
    }
    
  }
}
