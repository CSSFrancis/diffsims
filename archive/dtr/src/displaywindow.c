/*
 * displaywindow.c
 *
 * The display window
 *
 * (c) 2007-2009 Thomas White <taw27@cam.ac.uk>
 *
 *  dtr - Diffraction Tomography Reconstruction
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#define _GNU_SOURCE
#include <string.h>
#include <gtk/gtk.h>
#include <stdlib.h>
#include <math.h>
#include <gdk/gdkgl.h>
#include <gtk/gtkgl.h>
#include <glew.h>
#include <time.h>

#include "trackball.h"
#include "reflections.h"
#include "main.h"
#include "displaywindow.h"
#include "basis.h"
#include "dirax.h"
#include "reproject.h"
#include "cache.h"
#include "mapping.h"
#include "refine.h"
#include "imagedisplay.h"
#include "intensities.h"
#include "glbits.h"
#include "utils.h"

static gint displaywindow_changeview(GtkWidget *widget, GtkRadioAction *action,
					DisplayWindow *dw)
{
	GLfloat w = dw->drawing_area->allocation.width;
	GLfloat h = dw->drawing_area->allocation.height;
	dw->view = gtk_radio_action_get_current_value(action);

	if ( dw->view == DW_ORTHO ) {
		glbits_set_ortho(dw, w, h);
	} else {
		glbits_set_perspective(dw, w, h);
	}

	return 0;
}

static gint displaywindow_changemode(GtkWidget *widget, GtkRadioAction *action,
					DisplayWindow *dw)
{
	dw->mode = gtk_radio_action_get_current_value(action);
	displaywindow_update(dw);

	return 0;
}

static gboolean displaywindow_gl_button_press(GtkWidget *widget,
				GdkEventButton *event, DisplayWindow *dw)
{
	dw->x_start = event->x;
	dw->y_start = event->y;
	return FALSE;
}

static gint displaywindow_gl_motion_notify(GtkWidget *widget,
				GdkEventMotion *event, DisplayWindow *dw)
{
	float w = widget->allocation.width;
	float h = widget->allocation.height;
	float x = event->x;
	float y = event->y;
	float d_quat[4];
		
	struct timespec tim, tim2;
    tim.tv_sec = 0;
    tim.tv_nsec = 5000000;

	if ( event->state & GDK_CONTROL_MASK ) {

		/* Control-click changes 'zoom' */
		dw->distance += event->y - dw->y_start;
		if ( dw->distance < 1.0 ) dw->distance = 1.0;
		if ( dw->distance > 310.0 ) dw->distance = 310.0;
		if ( dw->view == DW_ORTHO ) {
			glbits_set_ortho(dw, w, h);
		} else {
			glbits_set_perspective(dw, w, h);
		}

	} else if ( event->state & GDK_SHIFT_MASK ) {

		/* Shift-click translates */
		dw->x_pos += (event->x - dw->x_start)/5;
		dw->y_pos += (event->y - dw->y_start)/5;

	} else {

		/* Click rotates */
		trackball(d_quat, (2.0*dw->x_start - w)/w,
				  (h-2.0*dw->y_start)/h,
				  (2.0*x-w)/w,
				  (h-2.0*y)/h);
				 
		add_quats(d_quat, dw->view_quat, dw->view_quat);
	    
		/*while (dw->view_quat[1] < -0.938) {
			
			nanosleep(&tim , &tim2);
			d_quat[0] = 0.000000;
			d_quat[1] = 0.001495;
			d_quat[2] = 0.000000;
			d_quat[3] = 0.999999;
			
			add_quats(d_quat, dw->view_quat, dw->view_quat);
			
			glbits_expose(dw->drawing_area, NULL, dw);
			
		}
		
		while (dw->view_quat[2] < 0.168) {
			
			nanosleep(&tim , &tim2);
			d_quat[0] = 0.001495;
			d_quat[1] = 0.000000;
			d_quat[2] = 0.000000;
			d_quat[3] = 0.999999;
			
			add_quats(d_quat, dw->view_quat, dw->view_quat);
			
			glbits_expose(dw->drawing_area, NULL, dw);
			
		}*/

	}

	dw->x_start = x;
	dw->y_start = y;

	gdk_window_invalidate_rect(widget->window, &widget->allocation, FALSE);

	return TRUE;
}

static gint displaywindow_closedown(GtkWidget *widget, DisplayWindow *dw)
{
	gtk_exit(0);
	return 0;
}

static void displaywindow_about(GtkWidget *widget, DisplayWindow *dw)
{
	GtkWidget *window;

	const gchar *authors[] = {
		"Thomas White <taw27@cam.ac.uk>",
		"Gordon Ball <gfb21@cam.ac.uk>",
		"Paul Midgley <pam33@cam.ac.uk>",
		NULL
	};

	window = gtk_about_dialog_new();

	gtk_about_dialog_set_name(GTK_ABOUT_DIALOG(window), PACKAGE_NAME);
	gtk_about_dialog_set_version(GTK_ABOUT_DIALOG(window), PACKAGE_VERSION);
	gtk_about_dialog_set_copyright(GTK_ABOUT_DIALOG(window),
				"(c) 2006-2009 Thomas White and contributors");
	gtk_about_dialog_set_comments(GTK_ABOUT_DIALOG(window),
				"Diffraction Tomography Reconstruction");
	gtk_about_dialog_set_license(GTK_ABOUT_DIALOG(window),
	  "(c) 2006-2008 Thomas White <taw27@cam.ac.uk>\n"
	  "Virtual trackball (c) Copyright 1993, 1994, Silicon Graphics, Inc.\n"
	  "See Credits for a full list of contributors\n"
	  "\n"
	  "Research funded by:\n"
	  "FEI Electron Optics B.V.\n"
	  "The Engineering and Physical Sciences Research Council");
	gtk_about_dialog_set_website(GTK_ABOUT_DIALOG(window),
	  "http://www-hrem.msm.cam.ac.uk/");
	gtk_about_dialog_set_authors(GTK_ABOUT_DIALOG(window), authors);

	g_signal_connect(window, "response", G_CALLBACK(gtk_widget_destroy),
				NULL);

	gtk_widget_show_all(window);
}

static void displaywindow_close(GtkWidget *widget, DisplayWindow *dw)
{
	gtk_exit(0);
}

static void displaywindow_dirax(GtkWidget *widget, DisplayWindow *dw)
{
	dirax_invoke(dw->ctx);
}

static void displaywindow_dirax_stop(GtkWidget *widget, DisplayWindow *dw)
{
	dirax_stop(dw->ctx);
}

static void displaywindow_dirax_rerun(GtkWidget *widget, DisplayWindow *dw)
{
	dirax_rerun(dw->ctx);
}

static void displaywindow_addui_callback(GtkUIManager *ui, GtkWidget *widget,
						GtkContainer *container)
{
	gtk_box_pack_start(GTK_BOX(container), widget, FALSE, FALSE, 0);

	/* Enable overflow menu if this is a toolbar */
	if ( GTK_IS_TOOLBAR(widget) ) {
		gtk_toolbar_set_show_arrow(GTK_TOOLBAR(widget), TRUE);
	}
}

static gint displaywindow_changecube(GtkWidget *widget, DisplayWindow *dw)
{
	dw->cube = gtk_toggle_action_get_active(GTK_TOGGLE_ACTION(widget));
	displaywindow_update(dw);
	return 0;
}

static gint displaywindow_changelines(GtkWidget *widget, DisplayWindow *dw)
{
	dw->lines = gtk_toggle_action_get_active(GTK_TOGGLE_ACTION(widget));
	displaywindow_update(dw);
	return 0;
}

static gint displaywindow_changebackground(GtkWidget *widget, DisplayWindow *dw)
{
	dw->background = gtk_toggle_action_get_active(
					GTK_TOGGLE_ACTION(widget) );
	displaywindow_update(dw);
	return 0;
}

static gint displaywindow_savecache_response(GtkWidget *widget, gint response,
						ControlContext *ctx)
{
	if ( response == GTK_RESPONSE_ACCEPT ) {
		char *cache_filename;
		cache_filename = gtk_file_chooser_get_filename(
					GTK_FILE_CHOOSER(widget) );
		if ( cache_save(ctx->images, cache_filename) ) {
			displaywindow_error("Failed to save cache file.",
						ctx->dw);
		}
		g_free(cache_filename);
	}

	gtk_widget_destroy(widget);

	return 0;
}

static gint displaywindow_savecache(GtkWidget *widget, DisplayWindow *dw)
{
	dw->savecache_window = gtk_file_chooser_dialog_new(
			"Save Image Analysis to Cache", GTK_WINDOW(dw->window),
					GTK_FILE_CHOOSER_ACTION_SAVE,
					GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
					GTK_STOCK_SAVE, GTK_RESPONSE_ACCEPT,
					NULL);
	g_signal_connect(G_OBJECT(dw->savecache_window), "response",
				G_CALLBACK(displaywindow_savecache_response),
				dw->ctx);
	gtk_widget_show_all(dw->savecache_window);

	return 0;
}

static gint displaywindow_setaxis_response(GtkWidget *widget, gint response,
						DisplayWindow *dw)
{
	if ( response == GTK_RESPONSE_OK ) {

		const char *offset;
		float off;

		offset = gtk_entry_get_text(GTK_ENTRY(dw->tiltaxis_entry));
		sscanf(offset, "%f", &off);

		mapping_adjust_axis(dw->ctx, deg2rad(off));

	}

	gtk_widget_destroy(widget);

	return 0;
}

static gint displaywindow_setaxis_activate(GtkWidget *widget,
						DisplayWindow *dw)
{
	return displaywindow_setaxis_response(dw->tiltaxis_window,
							GTK_RESPONSE_OK, dw);
}

static gint displaywindow_incraxis(GtkWidget *widget, DisplayWindow *dw)
{
	mapping_adjust_axis(dw->ctx, deg2rad(0.2));
	return 0;
}

static gint displaywindow_decraxis(GtkWidget *widget, DisplayWindow *dw)
{
	mapping_adjust_axis(dw->ctx, deg2rad(-0.2));
	return 0;
}

static gint displaywindow_setaxis(GtkWidget *widget, DisplayWindow *dw)
{

	GtkWidget *vbox;
	GtkWidget *hbox;
	GtkWidget *table;
	GtkWidget *label;

	dw->tiltaxis_window = gtk_dialog_new_with_buttons(
		"Set Tilt Axis Position", GTK_WINDOW(dw->window),
		GTK_DIALOG_DESTROY_WITH_PARENT,
		GTK_STOCK_CANCEL, GTK_RESPONSE_CLOSE,
		GTK_STOCK_OK, GTK_RESPONSE_OK,
		NULL);

	vbox = gtk_vbox_new(FALSE, 0);
	hbox = gtk_hbox_new(TRUE, 0);
	gtk_box_pack_start(GTK_BOX(GTK_DIALOG(dw->tiltaxis_window)->vbox),
					GTK_WIDGET(hbox), FALSE, FALSE, 7);
	gtk_box_pack_start(GTK_BOX(hbox), GTK_WIDGET(vbox), FALSE, FALSE, 5);

	table = gtk_table_new(1, 3, FALSE);
	gtk_table_set_row_spacings(GTK_TABLE(table), 5);
	gtk_table_set_col_spacings(GTK_TABLE(table), 5);
	gtk_box_pack_start(GTK_BOX(vbox), GTK_WIDGET(table), FALSE, FALSE, 0);

	label = gtk_label_new("Tilt Axis Offset:");
	gtk_misc_set_alignment(GTK_MISC(label), 1, 0.5);
	gtk_table_attach_defaults(GTK_TABLE(table), GTK_WIDGET(label),
							1, 2, 1, 2);
	label = gtk_label_new("degrees");
	gtk_table_attach_defaults(GTK_TABLE(table), GTK_WIDGET(label),
							3, 4, 1, 2);

	dw->tiltaxis_entry = gtk_entry_new();
	gtk_table_attach_defaults(GTK_TABLE(table),
				  GTK_WIDGET(dw->tiltaxis_entry), 2, 3, 1, 2);
	gtk_entry_set_alignment(GTK_ENTRY(dw->tiltaxis_entry), 1);

	g_signal_connect(G_OBJECT(dw->tiltaxis_window), "response",
			 G_CALLBACK(displaywindow_setaxis_response), dw);
	g_signal_connect(G_OBJECT(dw->tiltaxis_entry), "activate",
			 G_CALLBACK(displaywindow_setaxis_activate), dw);

	gtk_widget_show_all(dw->tiltaxis_window);
	gtk_widget_grab_focus(GTK_WIDGET(dw->tiltaxis_entry));

	return 0;

}

static gint displaywindow_image_first(GtkWidget *widget, DisplayWindow *dw)
{
	dw->cur_image = 0;
	displaywindow_update_imagestack(dw);

	return 0;
}

static gint displaywindow_image_prev(GtkWidget *widget, DisplayWindow *dw)
{

	if ( dw->cur_image > 0 ) {
		dw->cur_image--;
		displaywindow_update_imagestack(dw);
	}
	return 0;
}

static gint displaywindow_image_next(GtkWidget *widget, DisplayWindow *dw)
{
	if ( dw->cur_image < dw->ctx->images->n_images-1 ) {
		dw->cur_image++;
		displaywindow_update_imagestack(dw);
	}
	return 0;
}

static gint displaywindow_image_last(GtkWidget *widget, DisplayWindow *dw)
{
	dw->cur_image = dw->ctx->images->n_images-1;
	displaywindow_update_imagestack(dw);
	return 0;
}

static gint displaywindow_refinestep(GtkWidget *widget, DisplayWindow *dw) {
	refine_do_cell(dw->ctx);
	return 0;
}

static gint displaywindow_refinestack(GtkWidget *widget, DisplayWindow *dw)
{
	refine_do_sequence(dw->ctx);
	return 0;
}

static gint displaywindow_gl_destroyed(GtkWidget *widget, DisplayWindow *dw)
{
	glbits_final_free_resources(dw);
	return 0;
}

static gint displaywindow_extract(GtkWidget *widget, DisplayWindow *dw)
{
	GtkWidget *d;
	GtkAction *action;

	intensities_extract(dw->ctx);

	action = gtk_action_group_get_action(dw->action_group, "MappedAction");
	#ifdef HAVE_GTK_TEN
	gtk_radio_action_set_current_value(GTK_RADIO_ACTION(action),
						DW_MEASURED);
	#endif /* HAVE_GTK_TEN */
	displaywindow_update(dw);

	d = gtk_ui_manager_get_widget(dw->ui, "/ui/displaywindow/file/savehkl");
	gtk_widget_set_sensitive(d, TRUE);

	return 0;
}

static gint displaywindow_savehkl(GtkWidget *widget, DisplayWindow *dw)
{
	intensities_save(dw->ctx);
	return 0;
}

static gint displaywindow_loadcell(GtkWidget *widget, DisplayWindow *dw)
{
	basis_load(dw->ctx);
	return 0;
}

static gint displaywindow_savecell(GtkWidget *widget, DisplayWindow *dw)
{
	basis_save(dw->ctx);
	return 0;
}


static void displaywindow_addmenubar(DisplayWindow *dw)
{
	GtkActionEntry entries[] = {

	{ "FileAction", NULL, "_File", NULL, NULL, NULL },
	{ "SaveCacheAction", "filesave", "Save Image Analysis to _Cache",
			NULL, NULL, G_CALLBACK(displaywindow_savecache) },
	{ "SaveHKLAction", GTK_STOCK_SAVE, "Save Reflections...", NULL, NULL,
					G_CALLBACK(displaywindow_savehkl) },
	{ "LoadCellAction", GTK_STOCK_OPEN, "Load Unit Cell...", NULL, NULL,
					G_CALLBACK(displaywindow_loadcell) },
	{ "SaveCellAction", GTK_STOCK_SAVE, "Save Unit Cell...", NULL, NULL,
					G_CALLBACK(displaywindow_savecell) },
	{ "CloseAction", GTK_STOCK_QUIT, "_Quit", NULL, NULL,
					G_CALLBACK(displaywindow_close) },

	{ "ViewAction", NULL, "_View", NULL, NULL, NULL },
	
	{ "ToolsAction", NULL, "_Tools", NULL, NULL, NULL },
	{ "DirAxAction", "dtr-dirax", "Start _DirAx", "<Ctrl>D", NULL,
					G_CALLBACK(displaywindow_dirax) },
	{ "DirAxReRunAction", NULL, "Run another DirAx cycle", NULL, NULL,
					G_CALLBACK(displaywindow_dirax_rerun) },
	{ "StopDirAxAction", NULL, "Stop DirAx", NULL, NULL,
					G_CALLBACK(displaywindow_dirax_stop) },
	{ "RefineStepAction", "dtr-refine", "Refine Unit Cell", NULL, NULL,
					G_CALLBACK(displaywindow_refinestep) },
	{ "RefineSeqAction", "dtr-refineall", "Run Refinement Sequence",
			NULL, NULL, G_CALLBACK(displaywindow_refinestack) },
	{ "SetAxisAction", "dtr-tiltaxis", "Set Tilt Axis Position...", NULL,
				NULL, G_CALLBACK(displaywindow_setaxis) },
	{ "IncrAxisAction", NULL, "Increase Tilt Axis Position", "<Ctrl>Up",
				NULL, G_CALLBACK(displaywindow_incraxis) },
	{ "DecrAxisAction", NULL, "Decrease Tilt Axis Position",
			"<Ctrl>Down", NULL,
			G_CALLBACK(displaywindow_decraxis) },
	{ "ExtractIntensitiesAction", "dtr-quantify",
			"Quantify Reflection Intensities",
			NULL, NULL, G_CALLBACK(displaywindow_extract) },

	{ "HelpAction", NULL, "_Help", NULL, NULL, NULL },
	{ "AboutAction", GTK_STOCK_ABOUT, "_About DTR...", NULL, NULL,
			G_CALLBACK(displaywindow_about) },

	{ "ButtonDirAxAction", "dtr-dirax", "Run _DirAx", NULL, NULL,
			G_CALLBACK(displaywindow_dirax) },
	{ "ButtonTiltAxisAction", "dtr-tiltaxis", "Tilt Axis", NULL, NULL,
			G_CALLBACK(displaywindow_setaxis) },
	{ "ButtonRefineStepAction", "dtr-refine", "Refine", NULL, NULL,
			G_CALLBACK(displaywindow_refinestep) },
	{ "ButtonRefineSeqAction", "dtr-refineall", "Sequence", NULL, NULL,
			G_CALLBACK(displaywindow_refinestack) },
	{ "ButtonFirstImageAction", GTK_STOCK_GOTO_FIRST, "First Image",
			NULL, NULL, G_CALLBACK(displaywindow_image_first) },
	{ "ButtonPrevImageAction", GTK_STOCK_GO_BACK, "Previous Image",
			NULL, NULL, G_CALLBACK(displaywindow_image_prev) },
	{ "ButtonNextImageAction", GTK_STOCK_GO_FORWARD, "Next Image",
			NULL, NULL, G_CALLBACK(displaywindow_image_next) },
	{ "ButtonLastImageAction", GTK_STOCK_GOTO_LAST, "Last Image",
			NULL, NULL, G_CALLBACK(displaywindow_image_last) },
	{ "ButtonExtractIntensitiesAction", "dtr-quantify", "Quantify", NULL,
			NULL, G_CALLBACK(displaywindow_extract) },

	};
	guint n_entries = G_N_ELEMENTS(entries);

	GtkRadioActionEntry radios[] = {
		{ "OrthoAction", NULL, "_Orthographic Projection", NULL, NULL,
			DW_ORTHO },
		{ "PerspectiveAction", NULL, "_Perspective Projection", NULL,
			NULL, DW_PERSPECTIVE },
	};
	guint n_radios = G_N_ELEMENTS(radios);

	GtkRadioActionEntry radios2[] = {
		{ "MappedAction", NULL, "Show Mapped Features", NULL, NULL,
				DW_MAPPED },
		{ "MeasuredAction", NULL, "Show Measured Intensities", NULL,
				NULL, DW_MEASURED },
	};
	guint n_radios2 = G_N_ELEMENTS(radios2);

	GtkToggleActionEntry toggles[] = {
		{ "CubeAction", NULL, "Show 100 nm^-1 _Cube",
			NULL, NULL, G_CALLBACK(displaywindow_changecube),
					dw->cube },
		{ "LinesAction", NULL, "Show Indexing Lines",
			NULL, NULL, G_CALLBACK(displaywindow_changelines),
					dw->lines },
		{ "BackgroundAction", NULL, "Show Coloured Background",
			NULL, NULL, G_CALLBACK(displaywindow_changebackground),
					dw->background },
	};
	guint n_toggles = G_N_ELEMENTS(toggles);

	GError *error = NULL;

	dw->action_group = gtk_action_group_new("dtrdisplaywindow");
	gtk_action_group_add_actions(dw->action_group, entries, n_entries, dw);
	gtk_action_group_add_radio_actions(dw->action_group, radios, n_radios,
				-1, G_CALLBACK(displaywindow_changeview), dw);
	gtk_action_group_add_radio_actions(dw->action_group, radios2, n_radios2,
				-1, G_CALLBACK(displaywindow_changemode), dw);
	gtk_action_group_add_toggle_actions(dw->action_group, toggles,
				n_toggles, dw);

	dw->ui = gtk_ui_manager_new();
	gtk_ui_manager_insert_action_group(dw->ui, dw->action_group, 0);
	g_signal_connect(dw->ui, "add_widget",
			G_CALLBACK(displaywindow_addui_callback), dw->bigvbox);
	if ( gtk_ui_manager_add_ui_from_file(dw->ui,
			DATADIR"/dtr/displaywindow.ui", &error) == 0 ) {
		fprintf(stderr, "Error loading message window menu bar: %s\n",
				error->message);
		return;
	}

	gtk_window_add_accel_group(GTK_WINDOW(dw->window),
					gtk_ui_manager_get_accel_group(dw->ui));
	gtk_ui_manager_ensure_update(dw->ui);

}

/* Notify that something about the image stack display needs to be changed.
	eg. marks, tilt axis, image to be displayed. */
void displaywindow_update_imagestack(DisplayWindow *dw) {

	ImageFeatureList *flist;
	size_t j;
	ImageRecord *image;
	GtkWidget *d;

	/* Don't attempt to update the image stack if the stack doesn't exist */
	if ( dw->ctx->images->n_images == 0 ) return;

	double ang = dw->ctx->images->images[dw->cur_image].tilt;
	printf("The angle is %f\n", rad2deg(ang));

	imagedisplay_clear_marks(dw->stack);
	imagedisplay_put_data(dw->stack,
				dw->ctx->images->images[dw->cur_image]);

	image = &dw->ctx->images->images[dw->cur_image];

	if ( dw->ctx->cell_lattice ) {

		/* Perform relrod projection if necessary */
		if ( !image->rflist ) {
			image->rflist = reproject_get_reflections(image,
							dw->ctx->cell_lattice);
		}

		/* Draw the reprojected peaks */
		for ( j=0; j<image->rflist->n_features; j++ ) {
			imagedisplay_add_mark(dw->stack,
				image->rflist->features[j].x,
				image->rflist->features[j].y,
				IMAGEDISPLAY_MARK_CIRCLE_1,
				image->rflist->features[j].intensity);
		}

	}

	if ( image->features ) {
		/* Now draw the original measured peaks */
		flist = image->features;
		for ( j=0; j<flist->n_features; j++ ) {
			imagedisplay_add_mark(dw->stack, flist->features[j].x,
						flist->features[j].y,
						IMAGEDISPLAY_MARK_CIRCLE_2,
						flist->features[j].intensity);
		}
	}

	if ( image->features && dw->ctx->cell_lattice ) {
		/* Now connect partners */
		for ( j=0; j<image->rflist->n_features; j++ ) {
			if ( image->rflist->features[j].partner ) {
				imagedisplay_add_line(dw->stack,
					image->rflist->features[j].x,
					image->rflist->features[j].y,
					image->rflist->features[j].partner->x,
					image->rflist->features[j].partner->y,
					IMAGEDISPLAY_MARK_LINE_1);
			}
		}
	}

	if ( dw->cur_image == 0 ) {
		d = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindowtoolbar/first");
		gtk_widget_set_sensitive(GTK_WIDGET(d), FALSE);
		d = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindowtoolbar/prev");
		gtk_widget_set_sensitive(GTK_WIDGET(d), FALSE);
		d = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindowtoolbar/next");
		gtk_widget_set_sensitive(GTK_WIDGET(d), TRUE);
		d = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindowtoolbar/last");
		gtk_widget_set_sensitive(GTK_WIDGET(d), TRUE);
	} else if ( dw->cur_image == dw->ctx->images->n_images-1 ) {
		d = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindowtoolbar/first");
		gtk_widget_set_sensitive(GTK_WIDGET(d), TRUE);
		d = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindowtoolbar/prev");
		gtk_widget_set_sensitive(GTK_WIDGET(d), TRUE);
		d = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindowtoolbar/next");
		gtk_widget_set_sensitive(GTK_WIDGET(d), FALSE);
		d = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindowtoolbar/last");
		gtk_widget_set_sensitive(GTK_WIDGET(d), FALSE);
	} else {
		d = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindowtoolbar/first");
		gtk_widget_set_sensitive(GTK_WIDGET(d), TRUE);
		d = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindowtoolbar/prev");
		gtk_widget_set_sensitive(GTK_WIDGET(d), TRUE);
		d = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindowtoolbar/next");
		gtk_widget_set_sensitive(GTK_WIDGET(d), TRUE);
		d = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindowtoolbar/last");
		gtk_widget_set_sensitive(GTK_WIDGET(d), TRUE);
	}

}

static void displaywindow_switch_page(GtkNotebook *notebook,
					GtkNotebookPage *page, guint page_num,
					DisplayWindow *dw)
{
	if ( dw->realised && (page_num == 1) ) {
		int i;
		for ( i=0; i<2; i++ ) {
			glbits_expose(dw->drawing_area, NULL, dw);
		}
	}
}

static void displaywindow_scrolltoend(DisplayWindow *dw)
{
	GtkTextBuffer *buffer;
	GtkTextIter iter;

	buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(dw->messages));
	gtk_text_buffer_get_end_iter(buffer, &iter);
	gtk_text_iter_set_line_offset(&iter, 0);
	gtk_text_buffer_move_mark(buffer, dw->messages_mark, &iter);
	gtk_text_view_scroll_to_mark(GTK_TEXT_VIEW(dw->messages),
					dw->messages_mark, 0, TRUE, 1.0, 0.0);
}

static void displaywindow_message(DisplayWindow *dw, const char *text)
{
	GtkTextBuffer *buffer;
	GtkTextIter iter;

	buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(dw->messages));
	gtk_text_buffer_get_end_iter(buffer, &iter);
	gtk_text_buffer_insert_with_tags_by_name(buffer, &iter, text, -1,
						"default", NULL);

	displaywindow_scrolltoend(dw);
}

DisplayWindow *displaywindow_open(ControlContext *ctx)
{
	const char *filename;
	char *title;
	GdkGLConfig *glconfig;
	DisplayWindow *dw;
	GtkWidget *notebook;
	GtkWidget *d;
	GValue trueval = {0};

	dw = malloc(sizeof(DisplayWindow));
	
	filename =  basename(ctx->filename);
	title = malloc(10+strlen(filename));
	strcpy(title, filename);
	strcat(title, " - dtr");

	dw->ctx = ctx;
	ctx->dw = dw;

	dw->gl_use_buffers = 1;
	dw->gl_use_shaders = 1;
	dw->view = DW_ORTHO;
	dw->mode = DW_MAPPED;
	dw->distance = 150;
	dw->x_pos = 0;
	dw->y_pos = 0;
	dw->view_quat[0] = 0.0;
	dw->view_quat[1] = 1.0;
	dw->view_quat[2] = 0.0;
	dw->view_quat[3] = 0.0;
	dw->cube = TRUE;
	dw->lines = FALSE;
	dw->background = FALSE;
	dw->cur_image = 0;
	dw->realised = 0;

	dw->window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gtk_window_set_title(GTK_WINDOW(dw->window), title);
	free(title);
	dw->bigvbox = gtk_vbox_new(FALSE, 0);
	gtk_container_add(GTK_CONTAINER(dw->window), dw->bigvbox);
	displaywindow_addmenubar(dw);

	g_signal_connect(GTK_OBJECT(dw->window), "destroy",
				G_CALLBACK(displaywindow_closedown), dw);

	notebook = gtk_notebook_new();
	g_value_init(&trueval, G_TYPE_BOOLEAN);
	g_value_set_boolean(&trueval, TRUE);
	g_object_set_property(G_OBJECT(notebook), "homogeneous", &trueval);
	gtk_box_pack_end(GTK_BOX(dw->bigvbox), notebook, TRUE, TRUE, 0);
	/* Don't forget to update displaywindow_switch_page()
	 * when adding extra tabs */

	GtkWidget *messages_scroll;
	GdkColor colour;
	GtkTextBuffer *buffer;
	GtkTextIter iter;

	dw->messages = gtk_text_view_new();
	gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(dw->messages), GTK_WRAP_NONE);
	messages_scroll = gtk_scrolled_window_new(NULL, NULL);
	gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(messages_scroll),
				GTK_POLICY_AUTOMATIC, GTK_POLICY_ALWAYS);
	gtk_container_add(GTK_CONTAINER(messages_scroll),
				GTK_WIDGET(dw->messages));
	gtk_scrolled_window_set_shadow_type(
				GTK_SCROLLED_WINDOW(messages_scroll),
				GTK_SHADOW_IN);
	gdk_color_parse("white", &colour);
	gtk_widget_modify_bg(dw->messages, GTK_STATE_NORMAL, &colour);
	buffer = gtk_text_view_get_buffer(GTK_TEXT_VIEW(dw->messages));
	gtk_text_buffer_create_tag(buffer, "default", "left_margin", 3,
				"right_margin", 3, "font", "Mono", NULL);
	gtk_text_view_set_editable(GTK_TEXT_VIEW(dw->messages), FALSE);
	gtk_text_buffer_get_end_iter(buffer, &iter);
	dw->messages_mark = gtk_text_buffer_create_mark(buffer, NULL, &iter,
				FALSE);
	GTK_WIDGET_UNSET_FLAGS(dw->messages, GTK_CAN_FOCUS);
	gtk_notebook_append_page(GTK_NOTEBOOK(notebook), messages_scroll,
				gtk_label_new("Messages"));
	displaywindow_message(dw, "DTR version "PACKAGE_VERSION
			", (c) 2007-2009 Thomas White and contributors\n\n");

	/* GL stuff */
	glconfig = gdk_gl_config_new_by_mode(GDK_GL_MODE_RGB | GDK_GL_MODE_DEPTH
						 | GDK_GL_MODE_DOUBLE);
	if ( glconfig == NULL ) {
		fprintf(stderr, "Can't find double-buffered visual.\n");
		exit(1);
	}

	gtk_container_set_reallocate_redraws(GTK_CONTAINER(dw->window), TRUE);
	dw->drawing_area = gtk_drawing_area_new();
	gtk_widget_set_size_request(dw->drawing_area, 640, 640);
	gtk_widget_set_gl_capability(dw->drawing_area, glconfig, NULL, TRUE,
							GDK_GL_RGBA_TYPE);
	gtk_notebook_append_page(GTK_NOTEBOOK(notebook), dw->drawing_area,
					gtk_label_new("Reconstruction"));
	gtk_widget_add_events(dw->drawing_area,
				GDK_BUTTON1_MOTION_MASK | GDK_BUTTON_PRESS_MASK
					| GDK_VISIBILITY_NOTIFY_MASK);
	g_signal_connect(GTK_OBJECT(dw->drawing_area), "configure_event",
				G_CALLBACK(glbits_configure), dw);
	g_signal_connect(GTK_OBJECT(dw->drawing_area), "realize",
				G_CALLBACK(glbits_realise), dw);
	g_signal_connect(GTK_OBJECT(dw->drawing_area), "expose_event",
				G_CALLBACK(glbits_expose), dw);
	g_signal_connect(GTK_OBJECT(dw->drawing_area), "button_press_event",
				G_CALLBACK(displaywindow_gl_button_press), dw);
	g_signal_connect(GTK_OBJECT(dw->drawing_area), "motion_notify_event",
				G_CALLBACK(displaywindow_gl_motion_notify), dw);
	g_signal_connect(GTK_OBJECT(dw->drawing_area), "destroy",
				G_CALLBACK(displaywindow_gl_destroyed), dw);

	if ( ctx->images->n_images > 0 ) {
		dw->stack = imagedisplay_new_nowindow(
				ctx->images->images[dw->cur_image],
				IMAGEDISPLAY_SHOW_TILT_AXIS
				 | IMAGEDISPLAY_SHOW_CENTRE
		 		 | IMAGEDISPLAY_SCALE_BAR,
				NULL, NULL, NULL);
		gtk_notebook_append_page(GTK_NOTEBOOK(notebook),
					dw->stack->vbox,
					gtk_label_new("Image Stack"));
		displaywindow_update_imagestack(dw);
	} else {
		gtk_notebook_append_page(GTK_NOTEBOOK(notebook),
					gtk_label_new("No Images to Display"),
					gtk_label_new("Image Stack"));
	}

	g_signal_connect(GTK_OBJECT(notebook), "switch-page",
				G_CALLBACK(displaywindow_switch_page), dw);

	displaywindow_enable_cell_functions(dw, FALSE);
	displaywindow_update_dirax(ctx, dw);
	d = gtk_ui_manager_get_widget(dw->ui, "/ui/displaywindow/file/savehkl");
	gtk_widget_set_sensitive(d, FALSE);

	gtk_window_set_default_size(GTK_WINDOW(dw->window), 840, 800);
	gtk_widget_show_all(dw->window);
	//gtk_notebook_set_current_page(GTK_NOTEBOOK(notebook), 1);
	
	return dw;

}

void displaywindow_update(DisplayWindow *dw)
{
	glbits_free_resources(dw);
	glbits_prepare(dw);
	gdk_window_invalidate_rect(dw->drawing_area->window,
					&dw->drawing_area->allocation, FALSE);
}

/* Update the disabling of the menu for DirAx functions */
void displaywindow_update_dirax(ControlContext *ctx, DisplayWindow *dw)
{
	GtkWidget *start;
	GtkWidget *stop;
	GtkWidget *rerun;

	start = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindow/tools/dirax");
	stop = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindow/tools/diraxstop");
	rerun = gtk_ui_manager_get_widget(dw->ui,
					"/ui/displaywindow/tools/diraxrerun");

	if ( ctx->dirax ) {
		gtk_widget_set_sensitive(GTK_WIDGET(start), FALSE);
		gtk_widget_set_sensitive(GTK_WIDGET(stop), TRUE);
		gtk_widget_set_sensitive(GTK_WIDGET(rerun), TRUE);
	} else {
		gtk_widget_set_sensitive(GTK_WIDGET(start), TRUE);
		gtk_widget_set_sensitive(GTK_WIDGET(stop), FALSE);
		gtk_widget_set_sensitive(GTK_WIDGET(rerun), FALSE);
	}

}

void displaywindow_error(const char *msg, DisplayWindow *dw)
{
	GtkWidget *window;

	window = gtk_message_dialog_new(GTK_WINDOW(dw->window),
				GTK_DIALOG_DESTROY_WITH_PARENT,
				GTK_MESSAGE_WARNING, GTK_BUTTONS_CLOSE,
				"%s", msg);
	gtk_window_set_title(GTK_WINDOW(window), "Error");
	g_signal_connect_swapped(window, "response",
					G_CALLBACK(gtk_widget_destroy), window);
	gtk_widget_show(window);
}

void displaywindow_enable_cell_functions(DisplayWindow *dw, gboolean g)
{
	GtkWidget *d;

	d = gtk_ui_manager_get_widget(dw->ui,
				"/ui/displaywindow/tools/refinestep");
	gtk_widget_set_sensitive(d, g);
	d = gtk_ui_manager_get_widget(dw->ui,
				"/ui/displaywindow/tools/refineseq");
	gtk_widget_set_sensitive(d, g);
	d = gtk_ui_manager_get_widget(dw->ui,
				"/ui/displaywindowtoolbar/refineseq");
	gtk_widget_set_sensitive(d, g);
	d = gtk_ui_manager_get_widget(dw->ui,
				"/ui/displaywindowtoolbar/refinestep");
	gtk_widget_set_sensitive(d, g);
	d = gtk_ui_manager_get_widget(dw->ui,
				"/ui/displaywindow/tools/extractintensities");
	gtk_widget_set_sensitive(d, g);
	d = gtk_ui_manager_get_widget(dw->ui,
				"/ui/displaywindowtoolbar/extractintensities");
	gtk_widget_set_sensitive(d, g);
}
