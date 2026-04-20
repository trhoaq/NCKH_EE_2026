package com.nckh.voicecollector;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class RecordingAdapter extends RecyclerView.Adapter<RecordingAdapter.RecordingViewHolder> {
    private final List<RecordingItem> items = new ArrayList<>();

    public void submitList(List<RecordingItem> recordings) {
        items.clear();
        items.addAll(recordings);
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public RecordingViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_recording, parent, false);
        return new RecordingViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull RecordingViewHolder holder, int position) {
        RecordingItem item = items.get(position);
        String title = item.getNote().isEmpty() ? item.getOriginalFilename() : item.getNote();
        holder.titleTextView.setText(title);
        holder.metaTextView.setText(String.format(
                Locale.US,
                "ID %d | %s | %s",
                item.getId(),
                MainActivity.formatDuration(item.getDurationMs()),
                MainActivity.formatFileSize(item.getFileSizeBytes())
        ));
        holder.detailTextView.setText(String.format(
                Locale.US,
                "%s\n%s",
                item.getCreatedAt(),
                item.getFileUrl()
        ));
    }

    @Override
    public int getItemCount() {
        return items.size();
    }

    static class RecordingViewHolder extends RecyclerView.ViewHolder {
        final TextView titleTextView;
        final TextView metaTextView;
        final TextView detailTextView;

        RecordingViewHolder(@NonNull View itemView) {
            super(itemView);
            titleTextView = itemView.findViewById(R.id.recordingTitleTextView);
            metaTextView = itemView.findViewById(R.id.recordingMetaTextView);
            detailTextView = itemView.findViewById(R.id.recordingDetailTextView);
        }
    }
}
