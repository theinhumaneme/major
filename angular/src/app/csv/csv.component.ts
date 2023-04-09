import { Component, OnInit } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { ModelService } from '../model.service';
import { ContentObserver } from '@angular/cdk/observers';

interface Model {
  model: string;
  viewValue: string;
}

@Component({
  selector: 'app-csv',
  templateUrl: './csv.component.html',
  styleUrls: ['./csv.component.css'],
})
export class CsvComponent implements OnInit {
  blitz: FormGroup;
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
  constructor(private modelService: ModelService) {}

  ngOnInit(): void {
    this.initForm();
  }
  initForm(): void {
    let selected_model: String;

    let csv_values: String;

    this.blitz = new FormGroup({
      csv_values: new FormControl(csv_values, Validators.required),
      selected_model: new FormControl('', Validators.required),
    });
    this.blitz.controls['selected_model'].setValue(this.models[0].model, {
      onlySelf: true,
    });
  }
  onSubmit() {
    let res = this.blitz.value['csv_values'].split(',');
    let vals = [];
    for (var i = 0; i < res.length; i++) {
      vals.push(parseFloat(res[i]));
    }
    this.modelService.pred_model_array(vals, this.blitz.value['selected_model']).subscribe((resp) => {
      this.response = resp;
    });
  }
}
