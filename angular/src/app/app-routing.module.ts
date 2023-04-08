import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AppComponent } from './app.component';
import { FormComponent } from './form/form.component';
import { HomeComponent } from './home/home.component';
import { CsvComponent } from './csv/csv.component';

const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    redirectTo: 'app',
  },
  {
    path: 'app',
    component: HomeComponent,
  },
  {
    path: 'predict',
    component: FormComponent,
  },
  {
    path: 'blitz',
    component: CsvComponent,
  },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
