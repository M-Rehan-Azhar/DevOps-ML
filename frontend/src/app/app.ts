import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { FEATURES, Feature, SAMPLE_RECORD } from './feature-schema';
import { PredictionService, PredictionResponse } from './prediction.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  readonly fields = FEATURES;
  readonly deployedStatic = !['localhost', '127.0.0.1']
    .includes(window.location.hostname);
  values: Record<string, string | number | null> = {};
  // Signals keep the view in sync with HTTP callbacks (Angular 22 is zoneless).
  readonly loading = signal(false);
  readonly error = signal('');
  readonly result = signal<PredictionResponse | null>(null);
  private api = inject(PredictionService);

  constructor() {
    this.clear();
  }

  trackField(_index: number, field: Feature) { return field.name; }

  clear() {
    for (const field of this.fields) this.values[field.name] = null;
    this.error.set('');
    this.result.set(null);
  }

  fillSample() {
    this.values = { ...SAMPLE_RECORD };
    this.error.set('');
    this.result.set(null);
  }

  submit() {
    this.error.set(''); this.result.set(null);
    if (this.deployedStatic) {
      this.error.set('Static GitHub Pages demo: start Angular + Flask locally for predictions.');
      return;
    }
    const payload: Record<string, string | number> = {};
    for (const field of this.fields) {
      const value = this.values[field.name];
      if (value === null || value === undefined || value === '') {
        this.error.set(`Enter a value for ${field.label}`); return;
      }
      if (field.kind === 'number') {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
          this.error.set(`Invalid number for ${field.label}`); return;
        }
        if ((field.min !== undefined && numeric < field.min) ||
            (field.max !== undefined && numeric > field.max)) {
          this.error.set(`Out-of-range value for ${field.label} (allowed ${field.min} to ${field.max})`);
          return;
        }
        payload[field.name] = numeric;
      } else {
        payload[field.name] = String(value);
      }
    }
    this.loading.set(true);
    this.api.predict(payload).subscribe({
      next: result => { this.result.set(result); this.loading.set(false); },
      error: err => {
        this.error.set(`API error ${err.status || ''}: ` +
          (err.error?.error || err.message || 'Check Flask terminal'));
        this.loading.set(false);
      },
    });
  }
}
